# datara-ml

The ML pipeline for DATAra (ROADMAP track B): cleaning the data-harvester dataset, training the Global XGBoost next-hour model, and exporting it to ONNX.

So far: the B1 phantom-spike repair and the B2 feature contract (with a placeholder model and the app's parity fixture). Training (B3) is done by the team on top of the contract.

## Setup

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

Raw exports go in `data/`. That folder is git-ignored and must never be committed, because the respondent data is covered by RA 10173.

## B1: repair phantom spikes (harvester 1.6.7 and earlier)

### The problem

Up to version 1.6.7 (`schema_version` ≤ 3), the harvester could count a phone's lifetime mobile total as new usage in the first 30-second reading after mobile data came back up. That happened after Wi-Fi, airplane mode, no signal, or mobile data being switched off. Version 1.6.8 (`schema_version` 4) prevents it.

An investigation on 2026-11-05 found 48 such readings over 50 MB, across 11 devices. The largest was 27,637 MB in 30 seconds, in an hour where NetworkStatsManager measured 77.7 MB.

### The fix

The per-app NetworkStatsManager rows (`app_usage_records`, `network_type = 'MOBILE'`) come from Android's own accounting, not the counter that jumped. For every device and hour that contains rows with `schema_version` ≤ 3:

| Condition | Action |
|---|---|
| 30-second total > 1.5 × NSM total + 5 MB | use the NSM total (`replace_nsm`) |
| no NSM rows that hour, and a reconnect spike is present | drop the hour (`drop`) |
| otherwise | keep the 30-second total (`keep`) |

Hours made only of 1.6.8+ rows are never changed. The raw tables are never modified.

The rule works at the hour level, so it also catches phantom readings under 50 MB, for example on a phone restarted shortly before mobile data came back.

### Run it

1. Export both tables from the harvester's cloud database as CSV:
   - `SELECT * FROM usage_records`
   - `SELECT * FROM app_usage_records`
2. Save them to `data/`, then run:

```bash
python -m datara_ml.phantom_spikes --usage data/usage_records.csv --apps data/app_usage_records.csv --out data/hourly_mobile.csv --audit data/phantom_spike_audit.csv
```

The command writes two files and prints a summary:
- `hourly_mobile.csv`: one row per device and hour. `mobile_bytes` is the cleaned 30-second total and is empty for dropped hours.

**Since feature contract v2 the model's target no longer uses this step:** hourly usage comes from NetworkStatsManager (`nsm_hourly.py`), because the 30-second readings also miss time the app slept. B1 stays as evidence of the counter artifact; see [measurement gaps](docs/measurement-gaps.md).
- `phantom_spike_audit.csv`: every changed or dropped hour, with the 30-second and NSM totals in MB. Keep it as evidence for Chapter 3 and the defense.

### Reading the summary

- "replaced … without (smaller spikes)" counts hours replaced even though no reading was over 50 MB. If that number is large, the 30-second readings and NSM disagree for some other reason (for example an app NSM can't attribute). Check a few of those hours before training.
- Hour buckets use epoch hours. The Philippine offset (UTC+8) is a whole number of hours, so these are also local clock hours.

### Suggested Chapter 3 wording

> Readings recorded immediately after mobile data reconnected could contain the device's lifetime mobile counter, a known artifact of harvester versions ≤ 1.6.7. For each affected device-hour where the 30-second total exceeded 1.5 times the NetworkStatsManager total plus 5 MB, the NetworkStatsManager value was used; hours with no NetworkStatsManager record were excluded. *N* hours across *D* devices were corrected and *M* were excluded.

Fill in *N*, *D* and *M* from the printed summary.

## B2: feature contract

[`docs/feature-contract.md`](docs/feature-contract.md) (v2) fixes the 12 inputs of the Global XGBoost next-hour model, their order, missing-value rules and the ONNX signature. The Android app reproduces them exactly. [`datara_ml/features.py`](datara_ml/features.py) is the reference implementation:

```bash
python -m datara_ml.features --apps data/app_usage_records.csv --usage data/usage_records.csv --out data/train.csv
```

`data/train.csv` has `device_id`, `hour_start_ms`, the 12 features and `target` (log1p next-hour mobile bytes). Split by `device_id`.

**Internet speed is not a feature:** the harvester never recorded it.

**Hourly usage comes from NetworkStatsManager, not the 30-second readings**: they miss about a quarter of usage. [Measurement gaps](docs/measurement-gaps.md) explains why, with the queries ([`measurement-gaps.sql`](docs/measurement-gaps.sql)) and `python -m datara_ml.capture_report` to reproduce the numbers. See the contract's "What is not a feature".

## Placeholder model and parity fixture

```bash
python -m datara_ml.export_dev_model
```

Trains a tiny model on **synthetic** data and writes `artifacts/global_xgboost.onnx` (`model_version` `dev-synthetic-v3`) and `artifacts/golden_parity.json`. The app bundles both until the real model replaces them. To swap in a trained model, follow "Swapping in a trained model" in the contract.

## Training the Global model (v0 now, final on Nov 11)

```bash
python -m datara_ml.features --apps data/app_usage_records.csv --usage data/usage_records.csv --out data/train.csv
python -m datara_ml.train_v0 --train data/train.csv --model-version final-2026-11-11
```

Trains with the **Tweedie objective on raw next-hour bytes** (it predicts the mean usage of an hour), tunes `tweedie_variance_power`, `max_depth` and `n_estimators` by **nested** 5-fold cross-validation grouped by device (the reported scores were never used for tuning), compares it with two naive baselines on the same rows, and exports `artifacts/global_xgboost.onnx` (`output_transform = identity`), the parity fixture and `artifacts/v0_report.json` (aggregate metrics only). `--objective log1p_smear` trains the fallback: squared error on log1p with one smearing factor. Takes about a minute.

The bundled `v0-tweedie-2026-11-06` (contract v4, second 2026-11-06 export, 689 hours from 17 devices; tuned power 1.9, depth 2, 150 trees):

| Next hour, 17 devices held out | MAE (MB) | RMSE (MB) | Predicted ÷ actual total | Typical day-total error |
|---|---|---|---|---|
| **Tweedie** (`v0-tweedie-2026-11-06`) | **59.8** | **113.2** | **0.79** | 42% |
| log1p + smearing (fallback) | 63.1 | 119.8 | 0.82 | 51% |
| log1p, squared error (previous v0) | 59.1 | 120.3 | 0.43 | 68% |
| Baseline: last hour | 62.3 | 124.8 | 0.99 | 8% |
| Baseline: last 24 h average | 65.8 | 113.9 | 1.04 | 32% |

**Why Tweedie (decided 2026-11-06):** squared error on log1p predicts a typical hour, not the average one, so its predictions summed to only 43% of real usage, and the PET adds predicted hours, so its PETs came out late. Tweedie predicts the mean and has the lowest MAE and RMSE. It still underpredicts held-out devices (0.79), which the per-user local layer corrects with a usage ratio.

**Reading the baselines:** "last hour" looks best on day totals (8%) because it is scored one hour at a time with the real previous hour. Projecting a PET days ahead, it can only repeat one hour forever. Its log-scale MAE is also low because many hours are exactly zero. The fair comparison for the PET is B4's simulated cycles; on hourly error (MAE, RMSE) Tweedie beats both baselines.

## B4/E3 evaluation

[`datara_ml/evaluate.py`](datara_ml/evaluate.py) scores the model the way Chapter 4 reports it, with **whole devices held out**: the same 5 device folds as `train_v0`, the Global Tweedie model tuned and fitted inside each training fold, and every method scored on the same rows and cycles.

1. **Next hour**: MAE / RMSE (MB), R² and predicted ÷ actual total for the Global model and three baselines (last hour, average of the last 24 h, same hour yesterday).
2. **PET error (the thesis metric)** on simulated quota cycles: a cycle starts at every hour the app would show a forecast on a held-out device, with quotas of 100 MB, 500 MB, 1 GB and 3 GB. The projected PET is `features.project()`, the app's rule; the actual PET is when measured usage from that hour crossed the quota. Reported: MAE, median, share within ±3 h and ±24 h, and bias in hours (positive = late), for the model and the baselines projected the same way (last hour repeated, 24 h average repeated, yesterday's profile repeated).
3. **E3, Global vs Local**: the app's local layer (`LocalLayer.kt`: usage ratio per 6-hour block, 0.2–5×, switch after 168 learned hours when ≥ 5% better) replayed hour by hour on each held-out device.
4. Feature importance (gain) of the final model, and a sensitivity check with the final model's hyperparameters fixed in every fold.

**Which simulated cycles count.** Usage is added up hour by hour from the start hour. Hours inside NSM windows longer than 2 hours are unobserved, but the window's total is known, so they add their share of it and the running total is exact again at the window's end. A cycle is scored only when the crossing falls in an **observed** hour. It is skipped, and counted by reason in the report, when the crossing falls in an unobserved hour, an hour on the way has no NSM total at all, or the data ends first. A projection that never reaches the quota within 30 days is scored as start + 720 h (late).

Run it for the Nov 11 final model, **after** the two training commands above (same exports in `data/`):

```bash
python -m datara_ml.features --apps data/app_usage_records.csv --usage data/usage_records.csv --out data/train.csv
python -m datara_ml.train_v0 --train data/train.csv --model-version final-2026-11-11
python -m datara_ml.evaluate --apps data/app_usage_records.csv --usage data/usage_records.csv --train data/train.csv
```

It takes about 2 minutes and writes to `artifacts/eval/`: `report.json` (every number), `report.md` (the Chapter 4 tables) and the figures `next_hour_error.png`, `pet_mae_by_quota.png`, `pet_signed_error.png`, `pet_projected_vs_actual.png` and `feature_importance.png`. Devices appear only as D01, D02, … (in order of their first hour), so these files may be committed. `--train` only checks that `data/train.csv` was built from the same exports. For a quick run without tuning, pass `--fixed-params '{"tweedie_variance_power": 1.9, "max_depth": 2, "n_estimators": 150}'`.

**Preliminary result (second 2026-11-06 export, 689 hours, 17 devices; `artifacts/eval/`).** Next hour, the model matches `train_v0` (MAE 59.8 MB vs 62.3 for the last hour and 65.8 for the 24 h average). PET over 1,367 scored cycles from 13 devices: Global MAE 28.4 h (median 4.8 h, 79% within ±24 h, bias +3.2 h) vs 47.1 h for the 24 h average repeated and 246 h for the last hour repeated. Two caveats belong in Chapter 4:

- **The PET depends on the Tweedie power.** With the shipped hyperparameters (power 1.9) fixed in every fold, the PET MAE is 101 h, worse than the 24 h average (47 h), although its median (5.0 h) is still the best. The cause is the projection: predicted hours become the next hours' lags, and the model settles on a fixed rate. For some devices that rate is about 0 MB/h, usually after a run of measured-zero Wi-Fi hours, so the PET is "beyond 30 days" (186 of 1,367 cycles at power 1.9, 28 with the per-fold choices). The model was tuned on next-hour RMSE, which does not see this.
- **E3 cannot be shown yet.** No device has 168 measured hours, so the switch to Local never happens. In shadow, where the local layer was ready (≥ 24 learned hours, 9 devices), it did not beat Global: next-hour MAE 59.0 vs 51.8 MB and PET MAE 39.3 vs 38.6 h on the same rows. The new collection (from about 2026-10-30) should give some devices a full week.
