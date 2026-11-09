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
python -m datara_ml.train_v0 --train data/train.csv --apps data/app_usage_records.csv --usage data/usage_records.csv --model-version final-2026-11-11
```

Trains with the **Tweedie objective on raw next-hour bytes** (it predicts the mean usage of an hour), tunes `tweedie_variance_power`, `max_depth` and `n_estimators` by **nested** 5-fold cross-validation grouped by device (the reported scores were never used for tuning), choosing by **PET error** on B4's simulated quota cycles (see "B4/E3 evaluation"; next-hour RMSE only if `--apps`/`--usage` are left out), compares it with two naive baselines on the same rows, and exports `artifacts/global_xgboost.onnx` (`output_transform = identity`), the parity fixture and `artifacts/v0_report.json` (aggregate metrics only). `--objective log1p_smear` trains the fallback: squared error on log1p with one smearing factor. Takes about 2 minutes.

**Why PET error picks the hyperparameters (decided 2026-11-07):** the PET projection feeds each predicted hour back as the next hours' lags. Chosen by next-hour RMSE, the model had power 1.9, which settles on about 0 MB/h for some devices, usually after a run of measured-zero Wi-Fi hours. Its PETs then landed "beyond 30 days": 186 of 1,367 simulated cycles, and a PET MAE of 101 h, worse than repeating the 24 h average (47 h). Chosen by PET error, the model has power 1.3. None of its PETs land beyond 30 days, and its PET MAE is 14 h, with next-hour error unchanged.

The bundled `v0-tweedie-pet-2026-11-07` (contract v4, second 2026-11-06 export, 689 hours from 17 devices; chosen power 1.3, depth 2, 150 trees):

| Next hour, 17 devices held out | MAE (MB) | RMSE (MB) | Predicted ÷ actual total | Typical day-total error |
|---|---|---|---|---|
| **Tweedie, PET-selected** (`v0-tweedie-pet-2026-11-07`) | **59.8** | **111.4** | **0.81** | 46% |
| Tweedie, RMSE-selected (`v0-tweedie-2026-11-06`) | 59.8 | 113.2 | 0.79 | 42% |
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
4. **Forecast alerts as classification** (Chapter 3's classification metrics): the app's D8 alerts fire when the PET is within 24 h or 3 h. At every start hour and quota, each method's projection is a yes/no prediction ("reached within W hours"), scored against whether measured usage really reached the quota within W hours. Unlike the PET error, negatives count: a start that did *not* reach the quota is a true negative or a false alarm. Reported per horizon and quota: the confusion matrix (TP, FP, FN, TN), precision, recall, F1, accuracy and specificity, for the model and the baselines. A start is left out when the data cannot tell, e.g. a long NSM window holding the crossing straddles the boundary (counted as "not labelable").
5. Feature importance (gain) of the final model, and a sensitivity check with the final model's hyperparameters fixed in every fold.

**Which simulated cycles count.** Usage is added up hour by hour from the start hour. Hours inside NSM windows longer than 2 hours are unobserved, but the window's total is known, so they add their share of it and the running total is exact again at the window's end. A cycle is scored only when the crossing falls in an **observed** hour. It is skipped, and counted by reason in the report, when the crossing falls in an unobserved hour, an hour on the way has no NSM total at all, or the data ends first. A projection that never reaches the quota within 30 days is scored as start + 720 h (late).

Run it for the final model (Nov 11 rehearsal, Nov 12 final), **after** the two training commands above (same exports in `data/`):

```bash
python -m datara_ml.features --apps data/app_usage_records.csv --usage data/usage_records.csv --out data/train.csv
python -m datara_ml.train_v0 --train data/train.csv --apps data/app_usage_records.csv --usage data/usage_records.csv --model-version final-2026-11-11
python -m datara_ml.evaluate --apps data/app_usage_records.csv --usage data/usage_records.csv --train data/train.csv
```

Training takes about 2 minutes and the evaluation about 5 (more with more data). Hyperparameters are chosen by PET error in both, with every 3rd start hour scored (`PET_TUNE_STRIDE`). The evaluation writes to `artifacts/eval/`: `report.json` (every number), `report.md` (the Chapter 4 tables) and the figures `next_hour_error.png`, `pet_mae_by_quota.png`, `pet_signed_error.png`, `pet_projected_vs_actual.png`, `alert_confusion.png` (Global confusion matrices), `alert_scores.png` (precision / recall / F1 per method) and `feature_importance.png`. Devices appear only as D01, D02, … (in order of their first hour), so these files may be committed. `--train` only checks that `data/train.csv` was built from the same exports. For a quick run without tuning, pass `--fixed-params '{"tweedie_variance_power": 1.9, "max_depth": 2, "n_estimators": 150}'`.

**Preliminary result (second 2026-11-06 export, 689 hours, 17 devices; `artifacts/eval/`).** PET error is over 1,367 scored cycles from 13 devices (all quotas). Positive bias means late.

| | Next-hour MAE (MB) | PET MAE (h) | Median (h) | Within ±24 h | Bias (h) | Beyond 30 d |
|---|---|---|---|---|---|---|
| **Shipped model** (power 1.3, PET-selected), fixed in every fold | 58.8 | **13.8** | 4.2 | 81% | −10.9 | 0 |
| PET selection, nested (each fold tunes its own) | 59.7 | 15.6 | 4.7 | 78% | −9.8 | 0 |
| Before: RMSE selection, nested | 59.8 | 28.4 | 4.8 | 79% | +3.2 | 28 |
| Before: shipped power 1.9 (RMSE-selected), fixed in every fold | 56.2 | 101.1 | 5.0 | 74% | +87.5 | 186 |
| Baseline: average of last 24 h, repeated | 65.8 | 47.1 | 6.7 | 72% | +35.0 | 50 |
| Baseline: last hour, repeated | 62.3 | 246.4 | 22.7 | 51% | +239.6 | 450 |

**Forecast alerts (same run, nested, all quotas).** Within 24 h: 1,856 labelled starts, precision 0.75, recall 0.93, F1 0.83 (TP 1,093, FP 363, FN 82, TN 318); the 24 h average baseline also reaches F1 0.83 with higher precision (0.84) and lower recall (0.82). Within 3 h: 2,458 starts, precision 0.57, recall 0.73, F1 0.64, against F1 0.71 for the 24 h average. The model warns early, in line with its negative PET bias: few missed warnings, more false alarms. Its clear advantage stays the PET error.

Read the table with these in mind:

- **Use the nested row (15.6 h) as the unbiased number.** The shipped settings were chosen on the same devices they are scored on, so 13.8 h is slightly optimistic.
- **The PET-selected model errs early:** on average its PET comes about 10 h before the real one, and more for big quotas (−42 h at 3 GB). Early is the safer direction for a warning, but the error still grows with the quota (MAE 3 h at 100 MB, 50 h at 3 GB).
- **E3 cannot be shown yet.** No device has 168 measured hours, so the switch to Local never happens. Where the local layer was ready (≥ 24 learned hours, 9 devices), it ran in the background. Compared with Global on the same rows, it was worse one hour ahead (MAE 57.7 vs 50.4 MB) and about equal on PET (MAE 17.6 vs 18.2 h). The new collection (from about 2026-10-30) should give some devices a full week.
