"""Repair phantom mobile-usage spikes in data-harvester rows from app 1.6.7 and earlier.

Background
----------
Harvester versions up to 1.6.7 (``schema_version`` <= 3) could count a phone's lifetime
mobile-data total as usage in the first 30-second reading after mobile data came back up
(after Wi-Fi, airplane mode, no signal, or mobile data switched off). Version 1.6.8
(``schema_version`` 4) marks those readings ``COUNTER_RESET`` instead.

The per-app NetworkStatsManager rows (``app_usage_records``) come from Android's own
accounting, not the counter that jumped, so they serve as ground truth for the same hour.

Rule
----
For each device and clock hour, sum the 30-second cellular readings (TrafficStats, "TS")
and the per-app ``MOBILE`` NetworkStatsManager rows ("NSM"). Then, for hours that contain
any row with ``schema_version`` <= 3:

* TS > ``RATIO`` x NSM + ``MARGIN_BYTES``  -> use the NSM total     (action ``replace_nsm``)
* NSM missing and the hour holds a reconnect spike -> exclude hour  (action ``drop``)
* otherwise                                   -> keep the TS total  (action ``keep``)

Hours made only of ``schema_version`` >= 4 rows are always kept.

The raw tables are never modified; this produces a cleaned hourly table plus an audit
of every hour that was changed or dropped.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

HOUR_MS = 3_600_000

# Rows from harvester 1.6.7 and earlier can contain the artifact.
LAST_AFFECTED_SCHEMA = 3

# Hour-level rule: TS is implausible if it exceeds NSM by this much.
RATIO = 1.5
MARGIN_BYTES = 5_000_000  # 5 MB, absorbs small TS/NSM bucketing differences

# A single 30-second reading above this, right after a network change, is a spike suspect.
SPIKE_BYTES = 50_000_000  # 50 MB

NON_CELLULAR = {"WIFI", "NONE"}
# OK rows carry bytes; LEGACY rows are from 1.6.1, which had no status column.
COUNTED_STATUSES = {"OK", "LEGACY"}

USAGE_COLUMNS = ["device_id", "timestamp", "bytes_total", "network_type",
                 "measurement_status", "schema_version"]
APP_COLUMNS = ["device_id", "timestamp", "bytes_total", "network_type",
               "measurement_status"]


def _require(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def flag_reconnect_spikes(usage: pd.DataFrame) -> pd.DataFrame:
    """Return ``usage`` with ``prev_network_type`` and ``is_reconnect_spike`` columns.

    A reconnect spike is an OK cellular reading from schema <= 3 whose network type differs
    from the previous reading on the same device and whose bytes exceed ``SPIKE_BYTES``.
    This matches the SQL used in the 2026-11-05 investigation.
    """
    _require(usage, USAGE_COLUMNS, "usage_records")
    df = usage.sort_values(["device_id", "timestamp"], kind="stable").copy()
    df["prev_network_type"] = df.groupby("device_id")["network_type"].shift(1)
    cellular = ~df["network_type"].isin(NON_CELLULAR)
    changed = df["prev_network_type"].notna() & (df["network_type"] != df["prev_network_type"])
    df["is_reconnect_spike"] = (
        cellular
        & changed
        & (df["measurement_status"] == "OK")
        & (df["schema_version"] <= LAST_AFFECTED_SCHEMA)
        & (df["bytes_total"] > SPIKE_BYTES)
    )
    return df


def clean_hourly_mobile(usage: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Build hourly mobile totals per device with phantom spikes repaired.

    Returns one row per (device_id, hour_start_ms) that has at least one usage reading:

    ``ts_mobile_bytes``      sum of counted cellular 30-second readings
    ``nsm_mobile_bytes``     sum of per-app MOBILE NSM rows (NaN when the device has no
                             app rows at all in that hour)
    ``min_schema_version``   lowest schema_version among the hour's readings
    ``reconnect_spikes``     number of reconnect-spike readings in the hour
    ``action``               keep / replace_nsm / drop
    ``mobile_bytes``         cleaned hourly total (NaN for dropped hours)
    """
    _require(apps, APP_COLUMNS, "app_usage_records")
    df = flag_reconnect_spikes(usage)
    df["hour_start_ms"] = (df["timestamp"] // HOUR_MS) * HOUR_MS

    counted = df["measurement_status"].isin(COUNTED_STATUSES) & ~df["network_type"].isin(NON_CELLULAR)
    df["mobile_bytes_counted"] = np.where(counted, df["bytes_total"], 0)

    hourly = (
        df.groupby(["device_id", "hour_start_ms"], sort=True)
        .agg(
            ts_mobile_bytes=("mobile_bytes_counted", "sum"),
            min_schema_version=("schema_version", "min"),
            reconnect_spikes=("is_reconnect_spike", "sum"),
        )
        .reset_index()
    )

    a = apps.copy()
    a["hour_start_ms"] = (a["timestamp"] // HOUR_MS) * HOUR_MS
    a_counted = a["measurement_status"].isin(COUNTED_STATUSES) & (a["network_type"] == "MOBILE")
    a["mobile_bytes_counted"] = np.where(a_counted, a["bytes_total"], 0)
    # Any app row (MOBILE or WIFI) in the hour means NSM was read; then 0 MOBILE bytes is real.
    nsm = (
        a.groupby(["device_id", "hour_start_ms"])["mobile_bytes_counted"]
        .sum()
        .rename("nsm_mobile_bytes")
        .reset_index()
    )
    hourly = hourly.merge(nsm, on=["device_id", "hour_start_ms"], how="left")

    affected = hourly["min_schema_version"] <= LAST_AFFECTED_SCHEMA
    has_nsm = hourly["nsm_mobile_bytes"].notna()
    implausible = has_nsm & (
        hourly["ts_mobile_bytes"] > RATIO * hourly["nsm_mobile_bytes"].fillna(0) + MARGIN_BYTES
    )

    replace = affected & has_nsm & implausible
    drop = affected & ~has_nsm & (hourly["reconnect_spikes"] > 0)

    hourly["action"] = np.select([replace, drop], ["replace_nsm", "drop"], default="keep")
    hourly["mobile_bytes"] = np.select(
        [replace, drop],
        [hourly["nsm_mobile_bytes"], np.nan],
        default=hourly["ts_mobile_bytes"],
    )
    hourly["ts_mobile_bytes"] = hourly["ts_mobile_bytes"].astype("int64")
    hourly["reconnect_spikes"] = hourly["reconnect_spikes"].astype("int64")
    return hourly


def audit(hourly: pd.DataFrame) -> pd.DataFrame:
    """Only the hours that were changed or dropped, with MB columns for reading."""
    changed = hourly[hourly["action"] != "keep"].copy()
    changed["hour_start_utc"] = pd.to_datetime(changed["hour_start_ms"], unit="ms", utc=True)
    for col in ("ts_mobile_bytes", "nsm_mobile_bytes", "mobile_bytes"):
        changed[col.replace("_bytes", "_mb")] = (changed[col] / 1e6).round(3)
    return changed[[
        "device_id", "hour_start_ms", "hour_start_utc", "min_schema_version",
        "reconnect_spikes", "ts_mobile_mb", "nsm_mobile_mb", "mobile_mb", "action",
    ]]


def summary(hourly: pd.DataFrame) -> str:
    changed = hourly[hourly["action"] != "keep"]
    replaced = changed[changed["action"] == "replace_nsm"]
    lines = [
        f"hours total:                 {len(hourly)}",
        f"hours replaced with NSM:     {len(replaced)}",
        f"  with a reconnect spike:    {int((replaced['reconnect_spikes'] > 0).sum())}",
        f"  without (smaller spikes):  {int((replaced['reconnect_spikes'] == 0).sum())}",
        f"hours dropped (no NSM):      {int((changed['action'] == 'drop').sum())}",
        f"devices affected:            {changed['device_id'].nunique()}",
        f"spike readings found:        {int(hourly['reconnect_spikes'].sum())}",
        f"MB removed:                  "
        f"{(changed['ts_mobile_bytes'].sum() - changed['mobile_bytes'].fillna(0).sum()) / 1e6:,.1f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--usage", required=True, help="CSV export of usage_records")
    p.add_argument("--apps", required=True, help="CSV export of app_usage_records")
    p.add_argument("--out", required=True, help="cleaned hourly table (CSV)")
    p.add_argument("--audit", required=True, help="changed/dropped hours (CSV)")
    args = p.parse_args(argv)

    usage = pd.read_csv(args.usage, usecols=USAGE_COLUMNS)
    apps = pd.read_csv(args.apps, usecols=APP_COLUMNS)
    hourly = clean_hourly_mobile(usage, apps)
    hourly.to_csv(args.out, index=False)
    audit(hourly).to_csv(args.audit, index=False)
    print(summary(hourly))
    return 0


if __name__ == "__main__":
    sys.exit(main())
