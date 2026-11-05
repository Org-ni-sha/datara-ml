# Feature contract v1: Global XGBoost next-hour model

This is the fixed interface between training (this repo, B2–B5) and the Android app (`datara-android`, D6). The phone builds exactly these 12 values, in this order, from its own Room tables and feeds them to `global_xgboost.onnx`. A model trained on anything else will give plausible but wrong forecasts, so **any change here needs a new `CONTRACT_VERSION`, a matching app change, and a regenerated parity fixture.**

- Reference implementation: [`datara_ml/features.py`](../datara_ml/features.py)
- Phone implementation: `FeatureExtractor` and `PetProjector` in `datara-android` (`app/src/main/java/com/capstone/datara/forecast/`)
- Parity check: `artifacts/golden_parity.json`, written by [`export_dev_model.py`](../datara_ml/export_dev_model.py), replayed by `tests/test_export_dev_model.py` here and by `FeatureParityTest` in the app.

## Model interface

| | |
|---|---|
| ONNX input | name `features`, `tensor(float32)`, shape `[batch, 12]`. The phone always sends `[1, 12]`. |
| ONNX output | name `variable`, `tensor(float32)`, shape `[batch, 1]` |
| Output meaning | predicted `log1p(mobile bytes in the next hour)`. The app uses `max(0, expm1(output))`. |
| Missing values | `NaN` in the input. XGBoost learns a default branch for missing values, and the ONNX `TreeEnsembleRegressor` from onnxmltools follows it (`nodes_missing_value_tracks_true`). Tested: ONNX matches XGBoost within 1e-4 on rows with NaN, including an all-NaN row. |
| Metadata props | `contract_version` (`1`), `model_version` (logged in every `forecasts.model_version`), `feature_names` (comma-separated, the order below). The app refuses a model whose `contract_version` or feature count does not match. |
| Opset | `ai.onnx` 15 / `ai.onnx.ml` 3; runs on ONNX Runtime 1.31 (Python and Android). |

## Definitions used by every feature

- **Hour:** `[hour_start_ms, hour_start_ms + 3,600,000)`, epoch milliseconds. A 30-second reading belongs to the hour its end time falls in (harvester `timestamp`, phone `usage_samples.interval_end`).
- **Forecast hour `t`:** the hour being predicted. Features use only hours before `t`.
- **Observed hour:** an hour with at least one 30-second reading that B1 did not drop. Every value read from an unobserved hour is missing.
- **Mobile bytes of an hour:** the sum of `bytes_rx + bytes_tx` over readings whose status is `OK` or `LEGACY` and whose network type is not `WIFI` or `NONE` (the B1 rule in `phantom_spikes.py`). A measured 0 is observed, not missing.
- **Local time:** training uses **Philippine time, UTC+8**, the study's zone. The phone uses its own time zone, which for every participant is Asia/Manila; a phone set to another zone gets its own local hours, which is what the user's habits follow.
- **Signal reading is known:** `signal_strength`/`signal_dbm` is not `-999` and is negative.
- **Radio generation of a reading:** `2G`→2, `3G`→3, `LTE`→4, `5G`→5. `CELLULAR` (generation unknown), `WIFI`, `NONE` and anything else have none.
- All computation in float64, cast to float32 at the end.

## The 12 features

| # | Name | Unit | Definition | Missing when |
|---|---|---|---|---|
| 0 | `hour_sin` | – | `sin(2π · h / 24)`, `h` = local hour of `t` (0–23) | never |
| 1 | `hour_cos` | – | `cos(2π · h / 24)` | never |
| 2 | `day_of_week` | 0–6 | local weekday of `t`, Monday = 0 … Sunday = 6 | never |
| 3 | `is_weekend` | 0/1 | 1 when `day_of_week` ≥ 5 | never |
| 4 | `log_bytes_last_1h` | log1p(bytes) | mobile bytes of hour `t − 1h` | that hour is unobserved |
| 5 | `log_avg_bytes_last_3h` | log1p(bytes) | `log1p(mean)` of the mobile bytes of the observed hours among `t − 1h … t − 3h` | none of the 3 is observed |
| 6 | `log_avg_bytes_last_24h` | log1p(bytes) | same over `t − 1h … t − 24h` | none of the 24 is observed |
| 7 | `log_bytes_same_hour_yesterday` | log1p(bytes) | mobile bytes of hour `t − 24h` | unobserved |
| 8 | `log_bytes_same_hour_last_week` | log1p(bytes) | mobile bytes of hour `t − 168h` | unobserved |
| 9 | `signal_dbm_last_hour` | dBm | median of the known signal readings in hour `t − 1h` (an even count averages the middle two) | hour unobserved or no known reading |
| 10 | `radio_gen_last_hour` | 2–5 | the generation with the most readings in hour `t − 1h`; a tie goes to the higher generation | hour unobserved or no reading has a generation |
| 11 | `cellular_share_last_hour` | 0–1 | readings in `t − 1h` whose network type is not `WIFI`/`NONE`, divided by all its readings | hour unobserved |

Averages are of bytes, then `log1p` (not the mean of logs).

## Target

`log1p(mobile bytes of hour t)`, for observed hours only. Training rows also require **at least 6 observed hours among `t − 1h … t − 24h`** (`MIN_OBSERVED_LAST_24H`), the same rule the phone uses before it shows a forecast ("collecting" until then). Split train/test by `device_id`, never by row.

## PET projection (the app's rule, also for B4's simulated cycles)

`features.project()` is the reference; the phone's `PetProjector` reproduces it and the fixture checks it.

1. Start at forecast hour `t` with the cycle's remaining bytes `R`. If `R ≤ 0`, the PET is `t`.
2. For `k = 0, 1, …, 719` (30 days): build the features for hour `t + k`, predict bytes `b = max(0, expm1(model))`.
   - Byte lags (features 4–8) read real observed hours before `t` and the predicted bytes of hours `t … t+k−1`, which count as observed.
   - Network features (9–11): `k = 0` uses the real hour `t − 1h`. For `k ≥ 1` they are **pooled over the 24 hours before `t`**: the median of the observed hours' signal medians, the generation with the most summed readings (ties to the higher), and summed cellular readings over summed readings.
3. When cumulative predicted bytes reach `R` inside hour `t + k`, the PET is `t + k·1h + floor((R − cumulative_before) / b · 1h)`.
4. No reach within 720 hours: "beyond 30 days" (`reach_at = null`).

## What is not a feature, and why

| Candidate | Decision |
|---|---|
| **Internet speed** (mentioned in the manuscript) | **Excluded.** The harvester's `usage_records` has no speed or throughput column, so there is no training data for it. A bytes-per-second value derived from usage would just restate the usage features. The manuscript/protocol must be updated to say signal strength, radio generation and the share of time on cellular are the network inputs. |
| Screen on, battery level, charging | Collected by the harvester but not by DATAra's D4 telemetry; adding them needs new phone sampling. |
| Carrier / network operator | Recorded on both, but a categorical with few participants per carrier would let the model memorise people instead of learning patterns. Revisit only with many more devices. |
| Per-app usage | Phone-side NetworkStatsManager per-app reads are not part of DATAra's telemetry. |
| `cumulative_mb_today`, `time_period` | Derivable from the features above (24 h average, hour of day), so redundant. |

## Swapping in a trained model

The app has no model-specific code: a model trained on this contract drops in as a file swap.

1. Build the training set from the real exports:
   ```bash
   python -m datara_ml.phantom_spikes --usage data/usage_records.csv --apps data/app_usage_records.csv --out data/hourly_mobile.csv --audit data/phantom_spike_audit.csv
   python -m datara_ml.features --hourly data/hourly_mobile.csv --usage data/usage_records.csv --out data/train.csv
   ```
2. Train and tune your `XGBRegressor` on the 12 `FEATURES` columns (in that order, float32, NaN left as NaN — do not impute) with `target` as the label.
3. Export it with the contract's signature and a new version name:
   ```python
   from pathlib import Path
   from datara_ml.export_dev_model import export_onnx
   export_onnx(model, Path("artifacts/global_xgboost.onnx"), model_version="v0-2026-11-07")
   ```
   Check parity with `check_parity(model, OnnxPredictor(path), X_test)`.
4. Copy `artifacts/global_xgboost.onnx` to `datara-android/app/src/main/assets/models/global_xgboost.onnx`.
5. Regenerate the app's parity fixture against the new model and copy it too, so `FeatureParityTest` keeps checking the real file:
   ```bash
   python -m datara_ml.export_dev_model --fixture-only --model artifacts/global_xgboost.onnx --out artifacts
   ```
   then copy `artifacts/golden_parity.json` to `datara-android/app/src/test/resources/forecast/golden_parity.json`.
6. Run `gradlew test assembleDebug` in the app. No Kotlin change is needed unless the contract version changes.

## The placeholder model

`export_dev_model.py` (no arguments) trains a tiny model (60 trees, depth 4) on **synthetic** hourly usage and writes `artifacts/global_xgboost.onnx` with `model_version = dev-synthetic-v1`. It exists only so D6 can be built and tested before the real model; its forecasts mean nothing and must not be reported in the manuscript. Every forecast row it produces carries `model_version = dev-synthetic-v1`, so those rows can be excluded from E3.
