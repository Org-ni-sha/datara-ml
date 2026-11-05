"""Train the v0 Global model on the current (partial) harvester data and export it for the app.

v0 is an interim model: the dataset is a few days per device, so it is deliberately small and
untuned. The final model (after 14 days of collection) is trained by the team; this script is a
starting point, not the B3 tuning.

Evaluation is **grouped by device** (respondent): every fold holds out whole devices, so no person
is in both the training and the test rows (ROADMAP B3). Next-hour MAE is compared with two naive
baselines the model must beat:

* ``last_24h_avg``: next hour = average hourly usage over the last 24 h (feature 6)
* ``last_hour``:    next hour = usage in the last hour (feature 4; falls back to the 24 h average)

Usage::

    python -m datara_ml.train_v0 --train data/train.csv --model-version v0-2026-11-06

Writes ``artifacts/global_xgboost.onnx``, ``artifacts/golden_parity.json`` (the synthetic fixture
histories replayed against this model) and ``artifacts/v0_report.json`` (aggregate metrics only,
no device ids).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from datara_ml.export_dev_model import (
    OnnxPredictor, check_parity, export_onnx, fixture_history, golden_cases, write_fixture,
)
from datara_ml.features import FEATURES, PH_OFFSET_MS

PARAMS = dict(
    n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_lambda=5.0, random_state=0, n_jobs=1, tree_method="hist",
)
FOLDS = 5


def device_folds(devices: np.ndarray, k: int = FOLDS, seed: int = 0) -> list[np.ndarray]:
    """Split the unique devices into ``k`` groups (shuffled with a fixed seed)."""
    unique = np.array(sorted(set(devices)))
    np.random.default_rng(seed).shuffle(unique)
    return [unique[i::k] for i in range(k)]


def make_model(base_score: float):
    from xgboost import XGBRegressor
    return XGBRegressor(base_score=base_score, **PARAMS)


def baselines(x: pd.DataFrame) -> dict[str, np.ndarray]:
    """Naive log1p predictions. Missing values fall back to the 24 h average, then to 0."""
    avg24 = x["log_avg_bytes_last_24h"].fillna(0).to_numpy()
    last = x["log_bytes_last_1h"].fillna(x["log_avg_bytes_last_24h"]).fillna(0).to_numpy()
    return {"last_24h_avg": avg24, "last_hour": last}


def metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict[str, float]:
    y_mb = np.expm1(y_log) / 1e6
    p_mb = np.maximum(0, np.expm1(p_log)) / 1e6
    ss_res = float(np.sum((y_log - p_log) ** 2))
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    return {
        "mae_log": float(np.mean(np.abs(y_log - p_log))),
        "mae_mb": float(np.mean(np.abs(y_mb - p_mb))),
        "rmse_mb": float(math.sqrt(np.mean((y_mb - p_mb) ** 2))),
        "r2_log": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def cross_validate(train: pd.DataFrame) -> dict:
    """Out-of-fold predictions for every row, with whole devices held out per fold."""
    oof = np.full(len(train), np.nan)
    devices = train["device_id"].to_numpy()
    for held_out in device_folds(devices):
        test = np.isin(devices, held_out)
        fit = train[~test]
        model = make_model(float(fit["target"].mean()))
        model.fit(fit[FEATURES].to_numpy(np.float32), fit["target"].to_numpy(np.float32))
        oof[test] = model.predict(train.loc[test, FEATURES].to_numpy(np.float32))
    y = train["target"].to_numpy()
    report = {"model": metrics(y, oof)}
    for name, pred in baselines(train[FEATURES]).items():
        report[name] = metrics(y, pred)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train and export the v0 Global model.")
    p.add_argument("--train", required=True, help="training set from datara_ml.features")
    p.add_argument("--model-version", required=True, help="e.g. v0-2026-11-06")
    p.add_argument("--out", default="artifacts", help="output folder")
    args = p.parse_args(argv)
    out = Path(args.out)

    train = pd.read_csv(args.train)
    train[FEATURES] = train[FEATURES].astype(np.float32)
    cv = cross_validate(train)

    model = make_model(float(train["target"].mean()))
    x = train[FEATURES].to_numpy(np.float32)
    model.fit(x, train["target"].to_numpy(np.float32))
    onnx_path = out / "global_xgboost.onnx"
    export_onnx(model, onnx_path, args.model_version)
    predictor = OnnxPredictor(onnx_path)
    diff = check_parity(model, predictor, x)
    write_fixture(out / "golden_parity.json", args.model_version, PH_OFFSET_MS,
                  golden_cases(fixture_history(), predictor))

    gain = model.get_booster().get_score(importance_type="gain")
    importance = {FEATURES[int(k[1:])]: round(v, 4) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}
    report = {
        "model_version": args.model_version,
        "rows": len(train),
        "devices": int(train["device_id"].nunique()),
        "hours_from": int(train["hour_start_ms"].min()),
        "hours_to": int(train["hour_start_ms"].max()),
        "cv": f"{FOLDS}-fold grouped by device (out-of-fold, whole devices held out)",
        "params": PARAMS,
        "metrics": cv,
        "gain_importance": importance,
        "max_abs_xgboost_onnx": diff,
    }
    (out / "v0_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "devices", "metrics")}, indent=1))
    print("gain importance:", importance)
    print(f"max |xgboost - onnx|: {diff:.2e}; wrote {onnx_path}, golden_parity.json, v0_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
