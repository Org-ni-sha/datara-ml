"""How much mobile usage the 30-second TrafficStats rows capture, compared with NetworkStatsManager
over the same hours (docs/measurement-gaps.md). Prints aggregates and 8-character device prefixes.

    python -m datara_ml.capture_report --usage data/usage_records.csv --apps data/app_usage_records.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from datara_ml.nsm_hourly import nsm_hourly
from datara_ml.phantom_spikes import APP_COLUMNS, USAGE_COLUMNS, clean_hourly_mobile


def capture(usage: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Per device: NSM and spike-cleaned 30-second mobile MB over hours both measured."""
    nsm = nsm_hourly(apps).dropna(subset=["mobile_bytes"])
    ts = clean_hourly_mobile(usage[USAGE_COLUMNS], apps[APP_COLUMNS]).dropna(subset=["mobile_bytes"])
    both = nsm.merge(ts[["device_id", "hour_start_ms", "mobile_bytes"]], on=["device_id", "hour_start_ms"],
                     suffixes=("_nsm", "_ts"))
    out = both.groupby(both["device_id"].str[:8]).agg(
        hours=("hour_start_ms", "size"), nsm_mb=("mobile_bytes_nsm", "sum"), ts_mb=("mobile_bytes_ts", "sum"))
    out[["nsm_mb", "ts_mb"]] /= 1e6
    out = out[out["nsm_mb"] > 0]
    out["ts_pct_of_nsm"] = 100 * out["ts_mb"] / out["nsm_mb"]
    return out.rename_axis("device").sort_values("ts_pct_of_nsm").round(1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--usage", required=True)
    p.add_argument("--apps", required=True)
    args = p.parse_args(argv)
    table = capture(pd.read_csv(args.usage), pd.read_csv(args.apps))
    print(table.to_string())
    total = table[["nsm_mb", "ts_mb"]].sum()
    print(f"\nall devices: NSM {total.nsm_mb:,.0f} MB, 30-second {total.ts_mb:,.0f} MB "
          f"({100 * total.ts_mb / total.nsm_mb:.1f}%) over {int(table.hours.sum())} hours both measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
