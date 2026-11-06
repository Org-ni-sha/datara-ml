"""Train the Global next-hour model on the harvester data and export it for the app.

Objective (decided 2026-11-06): **Tweedie on raw next-hour bytes** (``reg:tweedie``). It predicts
the *mean* usage of an hour. The earlier squared error on log1p bytes predicted a typical hour,
which summed to only 43% of real usage, and the PET adds predicted hours together, so its PETs came
out late. Fallback (``--objective log1p_smear``): squared error on log1p with one Duan-style
smearing factor (total actual / total predicted bytes on the training rows).

Evaluation is **grouped by device** (respondent): whole devices are held out (ROADMAP B3).
Hyperparameters are tuned by **nested** cross-validation: inside each outer fold, a small grid is
scored with inner device folds on the training devices only, so the reported out-of-fold numbers
were never used for tuning. The grid is chosen by RMSE in bytes, which a mean predictor should
minimise. Every model is compared on the same rows with two naive baselines:

* ``last_hour``:    next hour = last hour's usage (falls back to the 24 h average)
* ``last_24h_avg``: next hour = average hourly usage over the last 24 h

Usage (the final model on Nov 11 is the same command with a new version)::

    python -m datara_ml.train_v0 --train data/train.csv --model-version v0-tweedie-2026-11-07

Writes ``artifacts/global_xgboost.onnx`` (``output_transform`` metadata: ``identity`` for Tweedie,
``expm1`` + ``output_scale`` for the fallback), ``artifacts/golden_parity.json`` and
``artifacts/v0_report.json`` (aggregate metrics only, no device ids).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from datara_ml.export_dev_model import (
    OnnxPredictor, check_parity, export_onnx, fixture_history, golden_cases, write_fixture,
)
from datara_ml.features import FEATURES, HOUR_MS, PH_OFFSET_MS

BASE = dict(subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
            random_state=0, n_jobs=1, tree_method="hist")
TWEEDIE_GRID = {
    "tweedie_variance_power": [1.1, 1.3, 1.5, 1.7, 1.9],
    "max_depth": [2, 3, 4],
    "n_estimators": [150, 300],
}
LOG1P_PARAMS = dict(n_estimators=150, max_depth=3, learning_rate=0.05, **BASE)
LEARNING_RATE = 0.05
FOLDS = 5
INNER_FOLDS = 4


def device_folds(devices: np.ndarray, k: int = FOLDS, seed: int = 0) -> list[np.ndarray]:
    """Split the unique devices into ``k`` groups (shuffled with a fixed seed)."""
    unique = np.array(sorted(set(devices)))
    np.random.default_rng(seed).shuffle(unique)
    return [unique[i::k] for i in range(k)]


# ---------------------------------------------------------------------------------------------
# Models: fit(X, y_bytes) -> object with .predict_bytes(X) and the export settings
# ---------------------------------------------------------------------------------------------

class TweedieModel:
    output_transform = "identity"
    output_scale = 1.0

    def __init__(self, **params):
        self.params = params

    def fit(self, x: np.ndarray, y_bytes: np.ndarray) -> "TweedieModel":
        from xgboost import XGBRegressor
        self.model = XGBRegressor(objective="reg:tweedie", learning_rate=LEARNING_RATE, **BASE, **self.params)
        self.model.fit(x, y_bytes.astype(np.float32))
        return self

    def predict_bytes(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).astype(np.float64)


class Log1pSmearModel:
    output_transform = "expm1"

    def __init__(self, **params):
        self.params = params or LOG1P_PARAMS

    def fit(self, x: np.ndarray, y_bytes: np.ndarray) -> "Log1pSmearModel":
        from xgboost import XGBRegressor
        y_log = np.log1p(y_bytes)
        self.model = XGBRegressor(base_score=float(y_log.mean()), **self.params)
        self.model.fit(x, y_log.astype(np.float32))
        raw = np.maximum(0, np.expm1(self.model.predict(x).astype(np.float64)))
        # Duan-style smearing, one factor: make predictions add up to the training total
        self.output_scale = float(y_bytes.sum() / raw.sum()) if raw.sum() > 0 else 1.0
        return self

    def predict_bytes(self, x: np.ndarray) -> np.ndarray:
        return self.output_scale * np.maximum(0, np.expm1(self.model.predict(x).astype(np.float64)))


def make(objective: str, **params):
    return TweedieModel(**params) if objective == "tweedie" else Log1pSmearModel(**params)


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------

def baselines(x: pd.DataFrame) -> dict[str, np.ndarray]:
    """Naive predictions in bytes. Missing values fall back to the 24 h average, then to 0."""
    avg24 = x["log_avg_bytes_last_24h"].fillna(0).to_numpy(np.float64)
    last = x["log_bytes_last_1h"].fillna(x["log_avg_bytes_last_24h"]).fillna(0).to_numpy(np.float64)
    return {"last_hour": np.expm1(last), "last_24h_avg": np.expm1(avg24)}


def metrics(rows: pd.DataFrame, y_bytes: np.ndarray, p_bytes: np.ndarray) -> dict[str, float]:
    """Error of byte predictions. ``rows`` gives device_id and hour_start_ms for daily totals."""
    p_bytes = np.maximum(0, p_bytes)
    y_mb, p_mb = y_bytes / 1e6, p_bytes / 1e6
    y_log, p_log = np.log1p(y_bytes), np.log1p(p_bytes)
    ss_tot = float(np.sum((y_log - y_log.mean()) ** 2))
    days = pd.DataFrame({"d": rows["device_id"].to_numpy(), "day": rows["hour_start_ms"].to_numpy() // (24 * HOUR_MS),
                         "y": y_bytes, "p": p_bytes}).groupby(["d", "day"])[["y", "p"]].sum()
    return {
        "mae_mb": float(np.mean(np.abs(y_mb - p_mb))),
        "rmse_mb": float(math.sqrt(np.mean((y_mb - p_mb) ** 2))),
        "mae_log": float(np.mean(np.abs(y_log - p_log))),
        "r2_log": 1 - float(np.sum((y_log - p_log) ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
        # The PET sums predictions hour by hour: below 1 means usage is underpredicted (PET late)
        "sum_pred_over_actual": float(p_bytes.sum() / y_bytes.sum()) if y_bytes.sum() > 0 else float("nan"),
        # Typical error of one device-day's total over the measured hours (days with >= 1 MB)
        "median_day_total_error": float(np.median(np.abs(days.p - days.y) / days.y.clip(lower=1e6))),
    }


# ---------------------------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------------------------

def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[FEATURES].to_numpy(np.float32), np.expm1(df["target"].to_numpy(np.float64))


def grid() -> list[dict]:
    keys = list(TWEEDIE_GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*TWEEDIE_GRID.values())]


def tune(train: pd.DataFrame, objective: str) -> dict:
    """Best grid point by RMSE (bytes) over inner device folds of ``train``."""
    if objective != "tweedie":
        return {}
    devices = train["device_id"].to_numpy()
    folds = device_folds(devices, k=min(INNER_FOLDS, len(set(devices))), seed=1)
    best, best_rmse = None, float("inf")
    for params in grid():
        sq, n = 0.0, 0
        for held in folds:
            te = np.isin(devices, held)
            x, y = _xy(train[~te])
            xt, yt = _xy(train[te])
            p = make(objective, **params).fit(x, y).predict_bytes(xt)
            sq += float(np.sum((yt - p) ** 2))
            n += len(yt)
        rmse = math.sqrt(sq / n)
        if rmse < best_rmse:
            best, best_rmse = params, rmse
    return best


def cross_validate(train: pd.DataFrame, objective: str = "tweedie") -> dict:
    """Nested grouped CV: out-of-fold byte predictions with tuning inside each outer fold."""
    oof = np.full(len(train), np.nan)
    devices = train["device_id"].to_numpy()
    chosen = []
    for held_out in device_folds(devices):
        te = np.isin(devices, held_out)
        params = tune(train[~te], objective)
        chosen.append(params)
        x, y = _xy(train[~te])
        oof[te] = make(objective, **params).fit(x, y).predict_bytes(train.loc[te, FEATURES].to_numpy(np.float32))
    y = np.expm1(train["target"].to_numpy(np.float64))
    report = {"model": metrics(train, y, oof)}
    for name, pred in baselines(train[FEATURES]).items():
        report[name] = metrics(train, y, pred)
    report["params_per_fold"] = chosen
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train and export the Global next-hour model.")
    p.add_argument("--train", required=True, help="training set from datara_ml.features")
    p.add_argument("--model-version", required=True, help="e.g. v0-tweedie-2026-11-07")
    p.add_argument("--objective", choices=["tweedie", "log1p_smear"], default="tweedie")
    p.add_argument("--out", default="artifacts", help="output folder")
    args = p.parse_args(argv)
    out = Path(args.out)

    train = pd.read_csv(args.train)
    train[FEATURES] = train[FEATURES].astype(np.float32)
    cv = cross_validate(train, args.objective)

    params = tune(train, args.objective)
    x, y = _xy(train)
    model = make(args.objective, **params).fit(x, y)
    onnx_path = out / "global_xgboost.onnx"
    export_onnx(model.model, onnx_path, args.model_version, model.output_transform, model.output_scale)
    predictor = OnnxPredictor(onnx_path)
    diff = check_parity(model.model, predictor, x)
    write_fixture(out / "golden_parity.json", args.model_version, PH_OFFSET_MS,
                  golden_cases(fixture_history(), predictor))

    gain = model.model.get_booster().get_score(importance_type="gain")
    importance = {FEATURES[int(k[1:])]: round(v, 4) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}
    report = {
        "model_version": args.model_version,
        "objective": args.objective,
        "output_transform": model.output_transform,
        "output_scale": model.output_scale,
        "rows": len(train),
        "devices": int(train["device_id"].nunique()),
        "hours_from": int(train["hour_start_ms"].min()),
        "hours_to": int(train["hour_start_ms"].max()),
        "cv": f"{FOLDS}-fold grouped by device, tuning nested in {INNER_FOLDS} inner device folds",
        "params": {**BASE, "learning_rate": LEARNING_RATE, **params} if args.objective == "tweedie" else LOG1P_PARAMS,
        "metrics": cv,
        "gain_importance": importance,
        "max_rel_xgboost_onnx": diff,
    }
    (out / "v0_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("objective", "rows", "devices", "params", "output_scale")}, indent=1))
    for name in ("model", "last_hour", "last_24h_avg"):
        m = cv[name]
        print(f"{name:13s} MAE {m['mae_mb']:6.1f} MB  RMSE {m['rmse_mb']:6.1f}  MAE_log {m['mae_log']:.2f}  "
              f"sum ratio {m['sum_pred_over_actual']:.2f}  day-total error {100 * m['median_day_total_error']:.0f}%")
    print(f"max relative |xgboost - onnx|: {diff:.2e}; wrote {onnx_path}, golden_parity.json, v0_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
