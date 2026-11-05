"""B2 feature contract: the inputs of the Global XGBoost next-hour model.

This is the **reference implementation** of ``docs/feature-contract.md``. The Android app
(D6, ``FeatureExtractor`` in ``datara-android``) reproduces it exactly, and the golden parity
fixture written by ``export_dev_model.py`` checks that it does. Change a definition here only
together with the contract, the app, and a new ``CONTRACT_VERSION``.

Inputs
------
* the cleaned hourly table from ``phantom_spikes.py`` (``hourly_mobile.csv``: ``device_id``,
  ``hour_start_ms``, ``mobile_bytes``; NaN = hour dropped by B1), and
* the raw ``usage_records`` export, from which :func:`hourly_network` builds the per-hour network
  statistics (reading counts, signal median, radio generation counts).

:func:`build_hourly_table` joins them. :func:`build_training_set` then produces one row per
observed device-hour: the 12 features (computed only from earlier hours) and the target.

Definitions
-----------
* An **hour** is ``[hour_start_ms, hour_start_ms + 1 h)`` in epoch milliseconds. A 30-second
  reading belongs to the hour its ``timestamp`` (the phone's ``interval_end``) falls in.
* An hour is **observed** when it has at least one reading and B1 did not drop it. Everything
  read from an unobserved hour is missing (NaN).
* **Mobile bytes** of an hour: the B1 rule (``phantom_spikes.COUNTED_STATUSES`` and a network
  type other than WIFI/NONE).
* Time features use **Philippine time (UTC+8)**, the study's zone. The phone uses its own zone,
  which for every participant is Asia/Manila.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from datara_ml.phantom_spikes import HOUR_MS, NON_CELLULAR

CONTRACT_VERSION = 1

# The ONNX input: one float32 row of N_FEATURES values, in this order.
FEATURES = [
    "hour_sin",                       # 0
    "hour_cos",                       # 1
    "day_of_week",                    # 2
    "is_weekend",                     # 3
    "log_bytes_last_1h",              # 4
    "log_avg_bytes_last_3h",          # 5
    "log_avg_bytes_last_24h",         # 6
    "log_bytes_same_hour_yesterday",  # 7
    "log_bytes_same_hour_last_week",  # 8
    "signal_dbm_last_hour",           # 9
    "radio_gen_last_hour",            # 10
    "cellular_share_last_hour",       # 11
]
N_FEATURES = len(FEATURES)
INPUT_NAME = "features"
OUTPUT_NAME = "variable"  # onnxmltools' default output name for a regressor

PH_OFFSET_MS = 8 * HOUR_MS
SIGNAL_UNKNOWN = -999

# Radio labels in the harvester's (and DATAra's RadioLabels) vocabulary -> generation.
# "CELLULAR" (unknown generation) and anything else count as no generation.
RADIO_GENERATIONS = {"2G": 2, "3G": 3, "LTE": 4, "5G": 5}
RADIO_COLUMNS = {2: "radio_2g", 3: "radio_3g", 4: "radio_lte", 5: "radio_5g"}

# The phone predicts only when at least this many of the 24 hours before the forecast hour are
# observed; build_training_set applies the same rule so the model trains on the rows it will see.
MIN_OBSERVED_LAST_24H = 6

# PET projection horizon (ROADMAP D6): beyond this the PET is "beyond 30 days".
PROJECTION_HORIZON_HOURS = 30 * 24

HOURLY_COLUMNS = ["device_id", "hour_start_ms", "mobile_bytes", "readings", "cellular_readings",
                  "signal_dbm", *RADIO_COLUMNS.values()]


# ---------------------------------------------------------------------------------------------
# Hourly statistics
# ---------------------------------------------------------------------------------------------

def is_known_signal(dbm: float) -> bool:
    """A real dBm reading: not the -999 sentinel and negative (0 or positive is a bad read)."""
    return dbm != SIGNAL_UNKNOWN and dbm < 0


def median(values: list[float]) -> float:
    """Median; an even count averages the middle two. NaN for no values."""
    if not values:
        return math.nan
    s = sorted(values)
    mid = len(s) // 2
    return float(s[mid]) if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def hourly_network(usage: pd.DataFrame) -> pd.DataFrame:
    """Per device-hour network statistics from raw ``usage_records`` (every reading counts).

    ``readings`` all readings; ``cellular_readings`` readings whose network type is not WIFI/NONE;
    ``signal_dbm`` median of the known ``signal_strength`` values (NaN if none);
    ``radio_2g`` … ``radio_5g`` readings per radio generation label.
    """
    needed = ["device_id", "timestamp", "network_type", "signal_strength"]
    missing = [c for c in needed if c not in usage.columns]
    if missing:
        raise ValueError(f"usage_records is missing columns: {missing}")
    df = usage[needed].copy()
    df["hour_start_ms"] = (df["timestamp"] // HOUR_MS) * HOUR_MS
    df["cellular"] = ~df["network_type"].isin(NON_CELLULAR)
    df["signal_known"] = df["signal_strength"].where(
        (df["signal_strength"] != SIGNAL_UNKNOWN) & (df["signal_strength"] < 0)
    )
    gen = df["network_type"].map(RADIO_GENERATIONS)
    for g, col in RADIO_COLUMNS.items():
        df[col] = (gen == g).astype("int64")
    out = (
        df.groupby(["device_id", "hour_start_ms"], sort=True)
        .agg(
            readings=("timestamp", "size"),
            cellular_readings=("cellular", "sum"),
            # pandas' median averages the middle two values, like median() above
            signal_dbm=("signal_known", "median"),
            **{col: (col, "sum") for col in RADIO_COLUMNS.values()},
        )
        .reset_index()
    )
    return out


def build_hourly_table(hourly_mobile: pd.DataFrame, network: pd.DataFrame) -> pd.DataFrame:
    """Join B1's cleaned hourly totals with :func:`hourly_network`."""
    hm = hourly_mobile[["device_id", "hour_start_ms", "mobile_bytes"]]
    out = hm.merge(network, on=["device_id", "hour_start_ms"], how="left")
    out["readings"] = out["readings"].fillna(0).astype("int64")
    out["cellular_readings"] = out["cellular_readings"].fillna(0).astype("int64")
    for col in RADIO_COLUMNS.values():
        out[col] = out[col].fillna(0).astype("int64")
    return out[HOURLY_COLUMNS].sort_values(["device_id", "hour_start_ms"]).reset_index(drop=True)


@dataclass(frozen=True)
class HourStats:
    """One observed hour. ``radio`` maps generation (2..5) to its reading count."""
    mobile_bytes: float
    readings: int = 0
    cellular_readings: int = 0
    signal_dbm: float = math.nan
    radio: tuple[int, int, int, int] = (0, 0, 0, 0)  # 2G, 3G, LTE, 5G


History = Mapping[int, HourStats]  # hour_start_ms -> stats; observed hours only


def device_histories(hourly: pd.DataFrame) -> dict[str, dict[int, HourStats]]:
    """Observed hours per device (B1-dropped hours, NaN ``mobile_bytes``, are left out)."""
    out: dict[str, dict[int, HourStats]] = {}
    for row in hourly.itertuples(index=False):
        if pd.isna(row.mobile_bytes):
            continue
        out.setdefault(row.device_id, {})[int(row.hour_start_ms)] = HourStats(
            mobile_bytes=float(row.mobile_bytes),
            readings=int(row.readings),
            cellular_readings=int(row.cellular_readings),
            signal_dbm=float(row.signal_dbm) if not pd.isna(row.signal_dbm) else math.nan,
            radio=(int(row.radio_2g), int(row.radio_3g), int(row.radio_lte), int(row.radio_5g)),
        )
    return out


# ---------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class NetworkFeatures:
    signal_dbm: float
    radio_gen: float
    cellular_share: float


def radio_gen_of(counts: tuple[int, int, int, int]) -> float:
    """The generation with the most readings; a tie goes to the higher one. NaN for none."""
    best, best_count = math.nan, 0
    for gen, count in zip((2, 3, 4, 5), counts):
        if count > 0 and count >= best_count:
            best, best_count = float(gen), count
    return best


def network_of_hour(stats: HourStats | None) -> NetworkFeatures:
    if stats is None:
        return NetworkFeatures(math.nan, math.nan, math.nan)
    share = stats.cellular_readings / stats.readings if stats.readings > 0 else math.nan
    return NetworkFeatures(stats.signal_dbm, radio_gen_of(stats.radio), share)


def pooled_network(history: History, forecast_hour_ms: int) -> NetworkFeatures:
    """The network over the 24 hours before ``forecast_hour_ms``, used for projected hours.

    signal: median of the observed hours' signal medians; radio: generation with the most summed
    readings; share: summed cellular readings / summed readings.
    """
    signals, radio, readings, cellular = [], [0, 0, 0, 0], 0, 0
    for j in range(1, 25):
        s = history.get(forecast_hour_ms - j * HOUR_MS)
        if s is None:
            continue
        if not math.isnan(s.signal_dbm):
            signals.append(s.signal_dbm)
        radio = [a + b for a, b in zip(radio, s.radio)]
        readings += s.readings
        cellular += s.cellular_readings
    share = cellular / readings if readings > 0 else math.nan
    return NetworkFeatures(median(signals), radio_gen_of(tuple(radio)), share)


def _log1p_or_nan(value: float) -> float:
    return math.log1p(value) if not math.isnan(value) else math.nan


def _bytes(history: History, hour_ms: int) -> float:
    s = history.get(hour_ms)
    return s.mobile_bytes if s is not None else math.nan


def _log_avg(history: History, forecast_hour_ms: int, hours: int) -> float:
    vals = [s.mobile_bytes for j in range(1, hours + 1)
            if (s := history.get(forecast_hour_ms - j * HOUR_MS)) is not None]
    return math.log1p(sum(vals) / len(vals)) if vals else math.nan


def observed_last_24h(history: History, forecast_hour_ms: int) -> int:
    return sum(1 for j in range(1, 25) if (forecast_hour_ms - j * HOUR_MS) in history)


def feature_vector(history: History, forecast_hour_ms: int, offset_ms: int = PH_OFFSET_MS,
                   network: NetworkFeatures | None = None) -> np.ndarray:
    """The 12 features for predicting the hour starting at ``forecast_hour_ms``.

    Only hours before it are read. ``network`` overrides the last-hour network features (the
    projection passes :func:`pooled_network` for hours after the first).
    """
    if forecast_hour_ms % HOUR_MS != 0:
        raise ValueError("forecast_hour_ms must be an hour boundary")
    local_ms = forecast_hour_ms + offset_ms
    hour = (local_ms // HOUR_MS) % 24
    # 1970-01-01 was a Thursday (3 when Monday = 0)
    dow = (local_ms // (24 * HOUR_MS) + 3) % 7
    angle = 2.0 * math.pi * hour / 24.0
    net = network if network is not None else network_of_hour(history.get(forecast_hour_ms - HOUR_MS))
    values = [
        math.sin(angle),
        math.cos(angle),
        float(dow),
        1.0 if dow >= 5 else 0.0,
        _log1p_or_nan(_bytes(history, forecast_hour_ms - HOUR_MS)),
        _log_avg(history, forecast_hour_ms, 3),
        _log_avg(history, forecast_hour_ms, 24),
        _log1p_or_nan(_bytes(history, forecast_hour_ms - 24 * HOUR_MS)),
        _log1p_or_nan(_bytes(history, forecast_hour_ms - 168 * HOUR_MS)),
        net.signal_dbm,
        net.radio_gen,
        net.cellular_share,
    ]
    return np.asarray(values, dtype=np.float64).astype(np.float32)


def build_training_set(hourly: pd.DataFrame, offset_ms: int = PH_OFFSET_MS,
                       min_observed_last_24h: int = MIN_OBSERVED_LAST_24H) -> pd.DataFrame:
    """One row per observed device-hour with enough history: ``device_id``, ``hour_start_ms``,
    the 12 features in contract order, and ``target`` = log1p(mobile bytes of that hour).

    Split by ``device_id`` (respondent), never by row (ROADMAP B3).
    """
    rows = []
    for device, history in device_histories(hourly).items():
        for hour_ms in sorted(history):
            if observed_last_24h(history, hour_ms) < min_observed_last_24h:
                continue
            x = feature_vector(history, hour_ms, offset_ms)
            rows.append([device, hour_ms, *x.tolist(), math.log1p(history[hour_ms].mobile_bytes)])
    cols = ["device_id", "hour_start_ms", *FEATURES, "target"]
    df = pd.DataFrame(rows, columns=cols)
    df[FEATURES] = df[FEATURES].astype(np.float32)
    return df


# ---------------------------------------------------------------------------------------------
# PET projection (the app's D6 rule, for B4's simulated cycles and the parity fixture)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Projection:
    next_hour_bytes: float
    reach_at_ms: int | None  # None = beyond the horizon


def predicted_bytes(log_prediction: float) -> float:
    """Model output (log1p bytes) to bytes, never negative."""
    return max(0.0, math.expm1(float(log_prediction)))


def project(history: History, forecast_hour_ms: int, remaining_bytes: float,
            predict_log: Callable[[np.ndarray], float], offset_ms: int = PH_OFFSET_MS,
            horizon_hours: int = PROJECTION_HORIZON_HOURS) -> Projection:
    """Step hour by hour from ``forecast_hour_ms``, subtracting each predicted hour from
    ``remaining_bytes``; the PET is interpolated linearly inside the hour that reaches zero.

    Predicted hours become observed hours for the next steps' byte lags. The first step uses the
    real last-hour network; later steps use :func:`pooled_network`.
    """
    hist = dict(history)
    pooled = pooled_network(history, forecast_hour_ms)
    cumulative = 0.0
    first = math.nan
    for k in range(horizon_hours):
        t = forecast_hour_ms + k * HOUR_MS
        x = feature_vector(hist, t, offset_ms, None if k == 0 else pooled)
        b = predicted_bytes(predict_log(x))
        if k == 0:
            first = b
            if remaining_bytes <= 0:
                return Projection(first, forecast_hour_ms)
        if b > 0 and cumulative + b >= remaining_bytes:
            fraction = (remaining_bytes - cumulative) / b
            return Projection(first, t + math.floor(fraction * HOUR_MS))
        cumulative += b
        hist[t] = HourStats(mobile_bytes=b)
    return Projection(first, None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the B2 training set (feature contract v1).")
    p.add_argument("--hourly", required=True, help="hourly_mobile.csv from phantom_spikes")
    p.add_argument("--usage", required=True, help="CSV export of usage_records")
    p.add_argument("--out", required=True, help="training set (CSV)")
    args = p.parse_args(argv)

    hourly_mobile = pd.read_csv(args.hourly)
    usage = pd.read_csv(args.usage, usecols=["device_id", "timestamp", "network_type", "signal_strength"])
    table = build_hourly_table(hourly_mobile, hourly_network(usage))
    train = build_training_set(table)
    train.to_csv(args.out, index=False)
    print(f"rows: {len(train)}  devices: {train['device_id'].nunique()}  features: {N_FEATURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
