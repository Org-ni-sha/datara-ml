"""Train a PLACEHOLDER Global model on synthetic data, export it to ONNX, and write the golden
parity fixture for the Android app (D6).

**The model this writes is not a research result.** It is trained on made-up hourly usage so the
app can be built and tested end to end before the real B3 model exists. It only exists to prove
that the app's features, ONNX inference and PET projection match this Python reference. Replace
it with the real model by running :func:`export_onnx` on your trained ``XGBRegressor`` (see
``docs/feature-contract.md``, "Swapping in a trained model").

Outputs (``--out``, default ``artifacts/``):

* ``global_xgboost.onnx``: input ``features`` float32 ``[batch, 12]``, output ``variable``
  float32 ``[batch, 1]`` = predicted log1p(next-hour mobile bytes). Metadata props:
  ``contract_version``, ``model_version``, ``feature_names``.
* ``golden_parity.json``: fixed hourly histories -> expected feature vectors -> expected ONNX
  outputs -> expected PET projections. Copy it to the app's test resources together with the
  model it was generated from.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from datara_ml.features import (
    CONTRACT_VERSION, FEATURES, INPUT_NAME, N_FEATURES, OUTPUT_NAME, PH_OFFSET_MS, HourStats,
    History, build_training_set, device_histories, feature_vector, project,
)
from datara_ml.phantom_spikes import HOUR_MS

DEV_MODEL_VERSION = "dev-synthetic-v3"
# 2026-11-02 00:00 Asia/Manila, an exact hour boundary in epoch ms
START_MS = 1_793_548_800_000
TARGET_OPSET = 15


# ---------------------------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------------------------

def synthetic_hourly(devices: int = 24, hours: int = 24 * 21, seed: int = 7) -> pd.DataFrame:
    """A made-up hourly table in ``build_hourly_table``'s shape: a daily usage curve per device,
    weekend boost, Wi-Fi hours with little mobile use, random gaps and unknown signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(devices):
        scale = rng.uniform(5e6, 60e6)              # this device's typical busy hour (bytes)
        peak = rng.uniform(18, 23)                  # local hour of its evening peak
        home_wifi = rng.uniform(0.2, 0.8)           # how often it sits on Wi-Fi at night
        gen_weights = rng.dirichlet([1, 2, 6, 2])   # 2G/3G/LTE/5G mix
        base_signal = rng.uniform(-110, -75)
        for k in range(hours):
            if rng.random() < 0.06:                 # phone off / tracking stopped
                continue
            hour_ms = START_MS + k * HOUR_MS
            local = ((hour_ms + PH_OFFSET_MS) // HOUR_MS) % 24
            dow = ((hour_ms + PH_OFFSET_MS) // (24 * HOUR_MS) + 3) % 7
            daily = math.exp(-((local - peak) ** 2) / 18) + 0.15 * math.exp(-((local - 12) ** 2) / 8)
            if local < 6:
                daily *= 0.05
            on_wifi = rng.random() < (home_wifi if (local >= 19 or local < 7) else 0.15)
            readings = int(rng.integers(100, 121))
            cellular = int(readings * (rng.uniform(0, 0.2) if on_wifi else rng.uniform(0.8, 1.0)))
            weekend = 1.4 if dow >= 5 else 1.0
            mobile = scale * daily * weekend * (cellular / readings) * rng.lognormal(0, 0.6)
            radio = rng.multinomial(cellular, gen_weights)
            signal = math.nan if rng.random() < 0.1 else float(round(base_signal + rng.normal(0, 6)))
            rows.append({
                "device_id": f"dev-{d:02d}", "hour_start_ms": hour_ms,
                "mobile_bytes": float(round(mobile)), "readings": readings,
                "cellular_readings": cellular, "signal_dbm": signal,
                "radio_2g": int(radio[0]), "radio_3g": int(radio[1]),
                "radio_lte": int(radio[2]), "radio_5g": int(radio[3]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Training and export
# ---------------------------------------------------------------------------------------------

def train_dev_model(train: pd.DataFrame):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=60, max_depth=4, learning_rate=0.15, subsample=0.9,
        # An explicit base_score: onnxmltools reads it from the parameters
        base_score=float(train["target"].mean()),
        random_state=0, n_jobs=1, tree_method="hist",
    )
    model.fit(train[FEATURES].to_numpy(np.float32), train["target"].to_numpy(np.float32))
    return model


def export_onnx(model, path: Path, model_version: str) -> None:
    """Convert a fitted ``XGBRegressor`` trained on the contract's 12 features to the ONNX file
    the app loads. Use this for the real model too; ``model_version`` ends up in every logged
    forecast (``forecasts.model_version``)."""
    import onnx
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    if getattr(model, "n_features_in_", N_FEATURES) != N_FEATURES:
        raise ValueError(f"model has {model.n_features_in_} features, the contract has {N_FEATURES}")
    onx = convert_xgboost(model, initial_types=[(INPUT_NAME, FloatTensorType([None, N_FEATURES]))],
                          target_opset=TARGET_OPSET)
    if onx.graph.output[0].name != OUTPUT_NAME:
        raise ValueError(f"unexpected output name {onx.graph.output[0].name!r}")
    for key, value in {
        "contract_version": str(CONTRACT_VERSION),
        "model_version": model_version,
        "feature_names": ",".join(FEATURES),
    }.items():
        prop = onx.metadata_props.add()
        prop.key, prop.value = key, value
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onx, str(path))


class OnnxPredictor:
    """log1p prediction for one feature row, as the app computes it (float32 in, float32 out)."""

    def __init__(self, path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def batch(self, x: np.ndarray) -> np.ndarray:
        out = self.session.run([OUTPUT_NAME], {INPUT_NAME: x.astype(np.float32)})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.batch(x.reshape(1, -1))[0])


def check_parity(model, predictor: OnnxPredictor, x: np.ndarray, atol: float = 1e-4) -> float:
    """Largest |XGBoost - ONNX| over the rows; raises if above ``atol``. Rows may contain NaN."""
    diff = float(np.max(np.abs(model.predict(x) - predictor.batch(x))))
    if diff > atol:
        raise AssertionError(f"XGBoost and ONNX disagree by {diff}")
    return diff


# ---------------------------------------------------------------------------------------------
# Golden fixture
# ---------------------------------------------------------------------------------------------

def _num(v: float) -> float | None:
    """JSON has no NaN: missing values are written as null."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _history_json(history: History) -> list[dict]:
    return [{
        "hour_start_ms": h, "mobile_bytes": s.mobile_bytes, "readings": s.readings,
        "cellular_readings": s.cellular_readings, "signal_dbm": _num(s.signal_dbm),
        "radio": list(s.radio),
    } for h, s in sorted(history.items())]


def golden_cases(history: History, predictor: OnnxPredictor) -> list[dict]:
    """A few forecast hours over one history: full history, a gap before the forecast hour,
    no last-week data, and a sparse history."""
    hours = sorted(history)
    last = hours[-1] + HOUR_MS
    week_ago = {h: s for h, s in history.items() if h >= last - 30 * HOUR_MS}
    gap = {h: s for h, s in history.items() if h < last - 3 * HOUR_MS}
    sparse = {h: s for i, (h, s) in enumerate(sorted(history.items())) if i % 3 == 0}
    # Earlier forecast hours whose previous hour has a known signal
    known = [h + HOUR_MS for h in hours[24 * 8:] if not math.isnan(history[h].signal_dbm)]
    mid = known[0]
    scenarios = [
        ("full_history", history, last),
        ("known_signal", {h: s for h, s in history.items() if h < mid}, mid),
        ("gap_before_forecast", gap, last),
        ("no_last_week", week_ago, last),
        ("sparse", sparse, last),
    ]
    cases = []
    for name, hist, t in scenarios:
        x = feature_vector(hist, t, PH_OFFSET_MS)
        projections = []
        for remaining in (0, 50_000_000, 1_000_000_000, 1_000_000_000_000):
            p = project(hist, t, remaining, predictor, PH_OFFSET_MS)
            projections.append({"remaining_bytes": remaining, "next_hour_bytes": p.next_hour_bytes,
                                "reach_at_ms": p.reach_at_ms})
        cases.append({
            "name": name,
            "history": _history_json(hist),
            "forecast_hour_ms": t,
            "expected_features": [_num(float(v)) for v in x],
            "expected_log_prediction": predictor(x),
            "projections": projections,
        })
    return cases


def nsm_parity_cases() -> list[dict]:
    """NSM windows -> hourly bytes, for the app's ``HourlyUsage`` (durable readings)."""
    from datara_ml.nsm_hourly import HOUR_MS as H, Window, hourly_from_windows

    t = START_MS
    m = 60_000
    scenarios = {
        # 10-minute reads over two hours, one straddling the hour boundary
        "contiguous_reads": [Window(t + i * 10 * m - 5 * m, t + (i + 1) * 10 * m - 5 * m, 1_000_000 + i * 7_919)
                             for i in range(13)],
        # a 90-minute GAP read is usable; its bytes are spread over both hours it touches
        "gap_under_2h": [Window(t, t + 30 * m, 4_000_000), Window(t + 30 * m, t + 120 * m, 9_000_000, "GAP")],
        # a 3-hour read makes every hour it touches unobserved, even the partly covered ones
        "window_over_2h": [Window(t, t + 50 * m, 2_000_000), Window(t + 50 * m, t + 230 * m, 30_000_000, "GAP"),
                           Window(t + 230 * m, t + 240 * m, 500_000)],
        # a counter reset makes its hour unobserved; a measured zero is observed
        "reset_and_zero": [Window(t, t + 60 * m, 0), Window(t + 60 * m, t + 70 * m, 0, "COUNTER_RESET"),
                           Window(t + 70 * m, t + 120 * m, 3_000_000)],
        # five uncovered minutes (tracking off) and an overlap both make their hour unobserved
        "uncovered_and_overlap": [Window(t, t + 25 * m, 1_000_000), Window(t + 30 * m, t + 60 * m, 1_000_000),
                                  Window(t + 60 * m, t + 100 * m, 2_000_000), Window(t + 90 * m, t + 120 * m, 2_000_000)],
        # a zero-byte read longer than 2 h is a measured zero for every hour it covers (v3)
        "long_zero_window": [Window(t, t + 20 * m, 1_000_000), Window(t + 20 * m, t + 200 * m, 0, "GAP"),
                             Window(t + 200 * m, t + 240 * m, 2_000_000)],
        # 30 seconds uncovered is within the tolerance
        "within_tolerance": [Window(t, t + 30 * m, 1_500_000), Window(t + 30 * m + 30_000, t + 60 * m, 1_500_000)],
    }
    cases = []
    for name, windows in scenarios.items():
        hourly = hourly_from_windows(windows)
        cases.append({
            "name": name,
            "windows": [{"start_ms": w.start_ms, "end_ms": w.end_ms, "mobile_bytes": w.mobile_bytes,
                         "status": w.status} for w in windows],
            "expected_hours": [{"hour_start_ms": h, "mobile_bytes": _num(b)} for h, b in sorted(hourly.items())],
        })
    return cases


def write_fixture(path: Path, model_version: str, offset_ms: int, cases: list[dict]) -> None:
    fixture = {
        "description": "Golden parity fixture for DATAra D6 (feature contract v%d). Generated by "
                       "datara_ml/export_dev_model.py from the model with the same model_version; "
                       "regenerate both together." % CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "model_version": model_version,
        "feature_names": FEATURES,
        "utc_offset_ms": offset_ms,
        "cases": cases,
        "nsm_cases": nsm_parity_cases(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def fixture_history() -> History:
    """The fixture's history: one held-out synthetic device (its own seed), 9 days."""
    return device_histories(synthetic_hourly(devices=1, hours=24 * 9, seed=2026))["dev-00"]


def model_version_of(path: Path) -> str:
    import onnx
    props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    if props.get("contract_version") != str(CONTRACT_VERSION):
        raise ValueError(f"{path} is for contract {props.get('contract_version')}, not {CONTRACT_VERSION}")
    return props["model_version"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export the synthetic PLACEHOLDER model and parity fixture.")
    p.add_argument("--out", default="artifacts", help="output folder")
    p.add_argument("--fixture-only", action="store_true",
                   help="only regenerate golden_parity.json for an existing model (--model)")
    p.add_argument("--model", help="the ONNX model for --fixture-only (e.g. your trained model)")
    args = p.parse_args(argv)
    out = Path(args.out)
    fixture_path = out / "golden_parity.json"

    if args.fixture_only:
        if not args.model:
            p.error("--fixture-only needs --model")
        model_path = Path(args.model)
        version = model_version_of(model_path)
        write_fixture(fixture_path, version, PH_OFFSET_MS, golden_cases(fixture_history(), OnnxPredictor(model_path)))
        print(f"wrote {fixture_path} for {model_path} ({version})")
        return 0

    hourly = synthetic_hourly()
    train = build_training_set(hourly)
    model = train_dev_model(train)
    onnx_path = out / "global_xgboost.onnx"
    export_onnx(model, onnx_path, DEV_MODEL_VERSION)
    predictor = OnnxPredictor(onnx_path)
    diff = check_parity(model, predictor, train[FEATURES].to_numpy(np.float32))
    write_fixture(fixture_path, DEV_MODEL_VERSION, PH_OFFSET_MS, golden_cases(fixture_history(), predictor))
    print(f"training rows: {len(train)}  max |xgboost - onnx|: {diff:.2e}")
    print(f"wrote {onnx_path} and {fixture_path} ({DEV_MODEL_VERSION}, PLACEHOLDER)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
