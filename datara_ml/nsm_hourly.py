"""Hourly mobile usage from NetworkStatsManager (NSM) windows: the target and byte lags of feature
contract v2. See ``docs/measurement-gaps.md`` for why NSM, not the 30-second TrafficStats rows.

Rule (reproduced by the app's ``HourlyUsage`` from ``durable_readings``):

* A **window** is one NSM read interval ``[start, end)`` with its mobile bytes. In the harvester,
  every ``app_usage_records`` row of one read shares ``start_time``/``end_time``; the window's
  mobile bytes are the sum of its ``MOBILE`` rows over **all** uids, including ``hotspot``
  (uid -5, tethered traffic) and other negative uids, so the total matches the device-level NSM
  query the phone uses (``querySummaryForDevice``, which also counts every uid).
* A window is **usable** when its status is in ``USABLE_STATUSES`` and it is at most
  ``MAX_WINDOW_MS`` (2 h) long, **or it has 0 mobile bytes** (contract v3): Android measured
  zero over the whole window, so every hour inside it is a measured zero whatever its length.
  Its bytes are spread evenly over its span and summed per clock hour.
* An hour is **observed** only when (1) every window touching it is usable, (2) those windows do
  not overlap, and (3) together they cover the hour, missing at most ``COVERAGE_TOLERANCE_MS``.
  Otherwise its bytes are unknown (missing for lags, excluded as a target). A measured 0 is
  observed.

Window times in ``app_usage_records`` are local clock times without a zone
(``2026-11-06T15:09:49``); they are converted with the row's ``utc_offset_minutes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000

# A window longer than this spreads bytes over too long a span to place them in hours.
MAX_WINDOW_MS = 2 * HOUR_MS
# Uncovered time an hour may have and still count as observed (window times are whole seconds).
COVERAGE_TOLERANCE_MS = 60_000
# NSM statuses whose bytes are real. GAP only means "longer than one read interval"; the
# 2-hour rule decides whether it can be used. LEGACY rows (harvester 1.6.1) have no status.
USABLE_STATUSES = frozenset({"OK", "GAP", "LEGACY"})

APP_COLUMNS = ["device_id", "start_time", "end_time", "network_type", "bytes_total",
               "measurement_status", "utc_offset_minutes"]


@dataclass(frozen=True)
class Window:
    start_ms: int
    end_ms: int
    mobile_bytes: float
    status: str = "OK"

    @property
    def usable(self) -> bool:
        span = self.end_ms - self.start_ms
        if self.status not in USABLE_STATUSES or span <= 0:
            return False
        # A zero-byte window is exact at any length: there is nothing to place in hours
        return span <= MAX_WINDOW_MS or self.mobile_bytes == 0


def hourly_from_windows(windows: Iterable[Window]) -> dict[int, float]:
    """Hour start (epoch ms) -> mobile bytes, for every hour a window touches. Unobserved hours
    map to NaN. Pure, so the app's implementation is checked against it (parity fixture)."""
    by_hour: dict[int, list[Window]] = {}
    for w in windows:
        if w.end_ms <= w.start_ms:
            if w.mobile_bytes > 0:  # bytes with no span: the hour can't be trusted
                by_hour.setdefault((w.start_ms // HOUR_MS) * HOUR_MS, []).append(w)
            continue
        h = (w.start_ms // HOUR_MS) * HOUR_MS
        while h < w.end_ms:
            by_hour.setdefault(h, []).append(w)
            h += HOUR_MS

    out: dict[int, float] = {}
    for hour, ws in by_hour.items():
        out[hour] = _hour_bytes(hour, ws)
    return out


def _hour_bytes(hour: int, ws: list[Window]) -> float:
    if any(not w.usable for w in ws):
        return float("nan")
    end_of_hour = hour + HOUR_MS
    covered = 0
    last_end = hour
    total = 0.0
    for w in sorted(ws, key=lambda w: (w.start_ms, w.end_ms)):
        s, e = max(w.start_ms, hour), min(w.end_ms, end_of_hour)
        if s < last_end:  # overlapping windows: which bytes belong here is ambiguous
            return float("nan")
        covered += e - s
        last_end = e
        total += w.mobile_bytes * (e - s) / (w.end_ms - w.start_ms)
    if HOUR_MS - covered > COVERAGE_TOLERANCE_MS:
        return float("nan")
    return total


def _local_to_epoch_ms(text: pd.Series, offset_minutes: pd.Series) -> pd.Series:
    local = pd.to_datetime(text, format="%Y-%m-%dT%H:%M:%S")
    epoch_local = (local - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    return (epoch_local - offset_minutes.astype("int64") * 60_000).astype("int64")


def windows_from_app_rows(apps: pd.DataFrame) -> pd.DataFrame:
    """One row per device and NSM read window: ``device_id``, ``start_ms``, ``end_ms``,
    ``mobile_bytes`` (all MOBILE rows, every uid) and ``status`` (the worst of its rows: any
    non-usable status wins)."""
    missing = [c for c in APP_COLUMNS if c not in apps.columns]
    if missing:
        raise ValueError(f"app_usage_records is missing columns: {missing}")
    df = apps[APP_COLUMNS].copy()
    df["start_ms"] = _local_to_epoch_ms(df["start_time"], df["utc_offset_minutes"])
    df["end_ms"] = _local_to_epoch_ms(df["end_time"], df["utc_offset_minutes"])
    df["mobile"] = np.where(df["network_type"] == "MOBILE", df["bytes_total"], 0)
    df["bad"] = ~df["measurement_status"].isin(USABLE_STATUSES)
    g = df.groupby(["device_id", "start_ms", "end_ms"], sort=True)
    out = g.agg(mobile_bytes=("mobile", "sum"), bad=("bad", "any"),
                status=("measurement_status", "first")).reset_index()
    first_bad = df[df["bad"]].groupby(["device_id", "start_ms", "end_ms"])["measurement_status"].first()
    out = out.set_index(["device_id", "start_ms", "end_ms"])
    out.loc[first_bad.index, "status"] = first_bad
    return out.reset_index()[["device_id", "start_ms", "end_ms", "mobile_bytes", "status"]]


def nsm_hourly(apps: pd.DataFrame) -> pd.DataFrame:
    """``device_id``, ``hour_start_ms``, ``mobile_bytes`` (NaN = unobserved) from the raw
    ``app_usage_records`` export."""
    windows = windows_from_app_rows(apps)
    rows = []
    for device, dw in windows.groupby("device_id", sort=True):
        ws = [Window(int(r.start_ms), int(r.end_ms), float(r.mobile_bytes), r.status)
              for r in dw.itertuples(index=False)]
        for hour, b in sorted(hourly_from_windows(ws).items()):
            rows.append((device, hour, b))
    return pd.DataFrame(rows, columns=["device_id", "hour_start_ms", "mobile_bytes"])
