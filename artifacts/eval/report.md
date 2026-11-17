# B4/E3 evaluation

2672 hours from 20 devices; every number is on devices held out of training (5-fold grouped by device (train_v0.device_folds, seed 0); Tweedie tuned inside each training fold (4 inner device folds)).

## Next-hour usage error

| Method | MAE (MB) | RMSE (MB) | R² | R² (log) | Predicted ÷ actual total |
|---|---|---|---|---|---|
| Global Tweedie model | 46.0 | 114.1 | 0.29 | -0.42 | 0.90 |
| Last hour | 48.3 | 135.3 | 0.01 | 0.71 | 0.99 |
| Average of last 24 h | 56.2 | 122.3 | 0.19 | 0.04 | 1.10 |
| Same hour yesterday | 68.1 | 161.1 | -0.41 | 0.06 | 1.18 |

2672 held-out hours.

## PET error on simulated quota cycles

Error = projected − actual PET in hours; positive = late. Cycles start at every forecast hour of a held-out device; see “Rules” below.

| Quota | Method | Cycles | Devices | MAE (h) | Median abs (h) | Within ±3 h | Within ±24 h | Bias (h) | Late | Beyond 30 d |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 MB | Global Tweedie model | 1393 | 14 | 5.7 | 1.2 | 68% | 93% | -4.8 | 39% | 0 |
| 100 MB | Last hour | 1393 | 14 | 225.3 | 4.5 | 45% | 63% | 223.2 | 70% | 428 |
| 100 MB | Average of last 24 h | 1393 | 14 | 64.6 | 1.8 | 60% | 84% | 58.0 | 52% | 111 |
| 100 MB | Same hour yesterday | 1393 | 14 | 65.1 | 2.2 | 56% | 84% | 58.8 | 54% | 111 |
| 500 MB | Global Tweedie model | 1016 | 14 | 9.8 | 4.9 | 39% | 88% | -5.9 | 46% | 0 |
| 500 MB | Last hour | 1016 | 14 | 268.8 | 25.1 | 26% | 50% | 263.9 | 75% | 367 |
| 500 MB | Average of last 24 h | 1016 | 14 | 86.7 | 8.4 | 31% | 71% | 76.1 | 57% | 94 |
| 500 MB | Same hour yesterday | 1016 | 14 | 87.1 | 9.7 | 30% | 70% | 75.6 | 56% | 94 |
| 1 GB | Global Tweedie model | 736 | 14 | 12.7 | 8.0 | 22% | 84% | -6.4 | 42% | 0 |
| 1 GB | Last hour | 736 | 14 | 285.5 | 43.8 | 16% | 43% | 278.3 | 74% | 278 |
| 1 GB | Average of last 24 h | 736 | 14 | 100.4 | 12.8 | 21% | 65% | 88.7 | 61% | 87 |
| 1 GB | Same hour yesterday | 736 | 14 | 100.7 | 13.4 | 21% | 64% | 88.5 | 60% | 87 |
| 3 GB | Global Tweedie model | 373 | 10 | 23.2 | 17.8 | 13% | 61% | -11.1 | 43% | 0 |
| 3 GB | Last hour | 373 | 10 | 312.7 | 142.8 | 5% | 26% | 295.6 | 72% | 155 |
| 3 GB | Average of last 24 h | 373 | 10 | 73.7 | 21.7 | 9% | 54% | 50.6 | 54% | 16 |
| 3 GB | Same hour yesterday | 373 | 10 | 73.7 | 22.7 | 9% | 54% | 50.1 | 52% | 16 |
| All | Global Tweedie model | 3518 | 15 | 10.2 | 4.0 | 44% | 86% | -6.1 | 42% | 0 |
| All | Last hour | 3518 | 15 | 259.7 | 21.7 | 29% | 51% | 254.2 | 72% | 1228 |
| All | Average of last 24 h | 3518 | 15 | 79.4 | 6.1 | 38% | 73% | 68.8 | 56% | 308 |
| All | Same hour yesterday | 3518 | 15 | 79.8 | 6.8 | 36% | 73% | 68.9 | 56% | 308 |

Device-averaged MAE (each device weighs the same, since heavy users give more cycles):

| Quota | Global Tweedie model | Last hour | Average of last 24 h | Same hour yesterday |
|---|---|---|---|---|
| 100 MB | 3.5 | 132.3 | 37.4 | 37.8 |
| 500 MB | 7.0 | 157.1 | 52.1 | 52.4 |
| 1 GB | 10.4 | 155.4 | 75.3 | 75.6 |
| 3 GB | 23.7 | 159.2 | 44.9 | 45.3 |
| All | 8.5 | 183.7 | 87.2 | 87.5 |

### Sensitivity: the final model's hyperparameters in every fold

Nested CV lets each fold pick its own hyperparameters. The shipped model uses `{"tweedie_variance_power": 1.5, "max_depth": 2, "n_estimators": 150}` (tuned on every device); the same held-out evaluation with those fixed in every fold gives next-hour MAE 45.8 MB and:

| Quota | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) | Beyond 30 d |
|---|---|---|---|---|---|---|
| 100 MB | 1393 | 5.6 | 1.2 | 94% | -4.6 | 0 |
| 500 MB | 1016 | 9.8 | 4.7 | 88% | -6.4 | 0 |
| 1 GB | 736 | 12.9 | 7.8 | 84% | -7.6 | 0 |
| 3 GB | 373 | 25.2 | 15.2 | 64% | -20.1 | 0 |
| All | 3518 | 10.4 | 4.0 | 87% | -7.4 | 0 |

### Which cycles could be scored

| Quota | Start hours | Scored | Crossing in an unobserved hour | Unknown hours before crossing | Never crossed in the data |
|---|---|---|---|---|---|
| 100 MB | 2672 | 1393 | 211 | 634 | 434 |
| 500 MB | 2672 | 1016 | 219 | 934 | 503 |
| 1 GB | 2672 | 736 | 199 | 1225 | 512 |
| 3 GB | 2672 | 373 | 157 | 1557 | 585 |

## E3: Global vs Local

Devices that learned ≥ 168 hours (the switch's minimum): **6** of 20; local layer ready (≥ 24 learned hours) on 15; switched to Local on **3** (309 hours shown Local).

| Next hour | Hours | MAE (MB) | RMSE (MB) | Predicted ÷ actual total |
|---|---|---|---|---|
| Global Tweedie model | 2672 | 46.0 | 114.1 | 0.90 |
| Local layer (as switched) | 2672 | 46.6 | 114.8 | 0.92 |
| Global, same rows/cycles | 2230 | 41.6 | 98.7 | 0.90 |
| Local layer (shadow, when ready) | 2230 | 43.8 | 101.2 | 1.03 |

| PET, all quotas | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) |
|---|---|---|---|---|---|
| Global Tweedie model | 3518 | 10.2 | 4.0 | 86% | -6.1 |
| Local layer (as switched) | 3518 | 10.1 | 3.9 | 86% | -6.7 |
| Global, same rows/cycles | 2895 | 10.5 | 4.7 | 87% | -5.8 |
| Local layer (shadow, when ready) | 2895 | 11.4 | 5.3 | 86% | -6.6 |

## Forecast alerts as classification

The app warns when the PET is within 24 h or within 3 h (D8). At every forecast hour and quota of a held-out device, each method's projection is a yes/no prediction, scored against whether measured usage really reached the quota within that time. Positive = reached within the horizon. Precision: when it warned, was it right? Recall: of the times the quota was about to be reached, how often did it warn?

### Within 24h (all quotas: 6626 labelled starts, 4062 not labelable)

Confusion matrix, Global model:

| | Actually reached | Not reached |
|---|---|---|
| **Alert** | 2842 (TP) | 1972 (FP) |
| **No alert** | 111 (FN) | 1701 (TN) |

| Quota | Method | Starts | Positives | Precision | Recall | F1 | Accuracy | Specificity |
|---|---|---|---|---|---|---|---|---|
| 100 MB | Global Tweedie model | 1985 | 70% | 0.70 | 1.00 | 0.83 | 0.70 | 0.00 |
| 100 MB | Last hour | 1985 | 70% | 0.96 | 0.66 | 0.78 | 0.74 | 0.94 |
| 100 MB | Average of last 24 h | 1985 | 70% | 0.89 | 0.90 | 0.90 | 0.85 | 0.73 |
| 100 MB | Same hour yesterday | 1985 | 70% | 0.89 | 0.90 | 0.90 | 0.85 | 0.73 |
| 500 MB | Global Tweedie model | 1683 | 54% | 0.54 | 0.97 | 0.69 | 0.54 | 0.02 |
| 500 MB | Last hour | 1683 | 54% | 0.87 | 0.57 | 0.69 | 0.72 | 0.90 |
| 500 MB | Average of last 24 h | 1683 | 54% | 0.76 | 0.81 | 0.78 | 0.76 | 0.70 |
| 500 MB | Same hour yesterday | 1683 | 54% | 0.76 | 0.81 | 0.78 | 0.76 | 0.70 |
| 1 GB | Global Tweedie model | 1509 | 38% | 0.52 | 0.90 | 0.66 | 0.65 | 0.49 |
| 1 GB | Last hour | 1509 | 38% | 0.75 | 0.55 | 0.63 | 0.76 | 0.89 |
| 1 GB | Average of last 24 h | 1509 | 38% | 0.69 | 0.73 | 0.71 | 0.78 | 0.80 |
| 1 GB | Same hour yesterday | 1509 | 38% | 0.69 | 0.73 | 0.71 | 0.78 | 0.80 |
| 3 GB | Global Tweedie model | 1449 | 5% | 0.24 | 0.61 | 0.34 | 0.88 | 0.89 |
| 3 GB | Last hour | 1449 | 5% | 0.24 | 0.53 | 0.33 | 0.89 | 0.91 |
| 3 GB | Average of last 24 h | 1449 | 5% | 0.33 | 0.70 | 0.45 | 0.91 | 0.92 |
| 3 GB | Same hour yesterday | 1449 | 5% | 0.33 | 0.70 | 0.45 | 0.91 | 0.92 |
| All | Global Tweedie model | 6626 | 45% | 0.59 | 0.96 | 0.73 | 0.69 | 0.46 |
| All | Last hour | 6626 | 45% | 0.84 | 0.61 | 0.70 | 0.77 | 0.91 |
| All | Average of last 24 h | 6626 | 45% | 0.78 | 0.84 | 0.81 | 0.82 | 0.81 |
| All | Same hour yesterday | 6626 | 45% | 0.78 | 0.84 | 0.81 | 0.82 | 0.81 |

### Within 3h (all quotas: 9663 labelled starts, 1025 not labelable)

Confusion matrix, Global model:

| | Actually reached | Not reached |
|---|---|---|
| **Alert** | 803 (TP) | 562 (FP) |
| **No alert** | 270 (FN) | 8028 (TN) |

| Quota | Method | Starts | Positives | Precision | Recall | F1 | Accuracy | Specificity |
|---|---|---|---|---|---|---|---|---|
| 100 MB | Global Tweedie model | 2453 | 32% | 0.61 | 0.90 | 0.73 | 0.79 | 0.74 |
| 100 MB | Last hour | 2453 | 32% | 0.78 | 0.70 | 0.74 | 0.84 | 0.91 |
| 100 MB | Average of last 24 h | 2453 | 32% | 0.60 | 0.82 | 0.69 | 0.77 | 0.74 |
| 100 MB | Same hour yesterday | 2453 | 32% | 0.59 | 0.71 | 0.65 | 0.75 | 0.77 |
| 500 MB | Global Tweedie model | 2394 | 10% | 0.46 | 0.43 | 0.44 | 0.90 | 0.95 |
| 500 MB | Last hour | 2394 | 10% | 0.46 | 0.49 | 0.47 | 0.90 | 0.94 |
| 500 MB | Average of last 24 h | 2394 | 10% | 0.52 | 0.52 | 0.52 | 0.91 | 0.95 |
| 500 MB | Same hour yesterday | 2394 | 10% | 0.41 | 0.52 | 0.46 | 0.88 | 0.92 |
| 1 GB | Global Tweedie model | 2394 | 3% | 0.33 | 0.02 | 0.03 | 0.97 | 1.00 |
| 1 GB | Last hour | 2394 | 3% | 0.24 | 0.30 | 0.26 | 0.96 | 0.98 |
| 1 GB | Average of last 24 h | 2394 | 3% | 0.29 | 0.32 | 0.30 | 0.96 | 0.98 |
| 1 GB | Same hour yesterday | 2394 | 3% | 0.25 | 0.30 | 0.27 | 0.96 | 0.98 |
| 3 GB | Global Tweedie model | 2422 | 0% | – | – | – | 1.00 | 1.00 |
| 3 GB | Last hour | 2422 | 0% | 0.00 | – | – | 1.00 | 1.00 |
| 3 GB | Average of last 24 h | 2422 | 0% | – | – | – | 1.00 | 1.00 |
| 3 GB | Same hour yesterday | 2422 | 0% | 0.00 | – | – | 1.00 | 1.00 |
| All | Global Tweedie model | 9663 | 11% | 0.59 | 0.75 | 0.66 | 0.91 | 0.93 |
| All | Last hour | 9663 | 11% | 0.66 | 0.63 | 0.65 | 0.92 | 0.96 |
| All | Average of last 24 h | 9663 | 11% | 0.57 | 0.73 | 0.64 | 0.91 | 0.93 |
| All | Same hour yesterday | 9663 | 11% | 0.53 | 0.64 | 0.58 | 0.90 | 0.93 |


## Feature importance (gain, final model)

| Feature | Gain share |
|---|---|
| `log_bytes_last_1h` | 39% |
| `log_avg_bytes_last_3h` | 24% |
| `log_avg_bytes_last_24h` | 11% |
| `hour_cos` | 4% |
| `log_bytes_same_hour_yesterday` | 4% |
| `log_bytes_same_hour_last_week` | 4% |
| `hour_sin` | 3% |
| `cellular_share_last_hour` | 3% |
| `signal_dbm_last_hour` | 3% |
| `radio_gen_last_hour` | 3% |
| `day_of_week` | 2% |
| `is_weekend` | 0% |

## Per device (anonymous)

Counts are the app's hourly runs that made a forecast (including hours never measured afterwards); “learned” counts the measured hours the local layer learned from.

| Device | Fold | Forecast runs | Hours learned | Runs with local ready | Runs shown Local |
|---|---|---|---|---|---|
| D01 | 1 | 452 | 408 | 412 | 0 |
| D02 | 2 | 358 | 274 | 296 | 0 |
| D03 | 4 | 384 | 260 | 337 | 0 |
| D04 | 1 | 395 | 294 | 363 | 155 |
| D05 | 3 | 320 | 295 | 293 | 130 |
| D06 | 4 | 93 | 45 | 26 | 0 |
| D07 | 3 | 58 | 31 | 8 | 0 |
| D08 | 3 | 338 | 158 | 266 | 0 |
| D09 | 5 | 227 | 71 | 150 | 0 |
| D10 | 4 | 78 | 18 | 0 | 0 |
| D11 | 5 | 67 | 16 | 0 | 0 |
| D12 | 2 | 154 | 151 | 130 | 0 |
| D13 | 5 | 137 | 45 | 41 | 0 |
| D14 | 3 | 67 | 16 | 0 | 0 |
| D15 | 2 | 66 | 19 | 0 | 0 |
| D16 | 2 | 52 | 13 | 0 | 0 |
| D17 | 4 | 291 | 159 | 243 | 0 |
| D18 | 1 | 227 | 117 | 181 | 0 |
| D19 | 5 | 240 | 224 | 216 | 57 |
| D20 | 1 | 112 | 58 | 69 | 0 |

## Rules

- **start_hours**: every training row of a held-out device (observed hour, >= 6 observed of the last 24)
- **quotas_mb**: [100, 500, 1000, 3000]
- **actual_pet**: measured cumulative from the start hour; long NSM windows add their spread total; scored only if the crossing falls in an observed hour
- **error_sign**: projected - actual in hours; positive = late
- **beyond_horizon**: a projection that never reaches the quota in 720 h is scored as start + 720 h
- **alerts**: every start hour and quota; predicted = the projection reaches the quota within W h (AlertRules: 24 h, 3 h); actual = measured usage reached it within W h; starts the data cannot label (a long NSM window across the boundary, a hole, or the data ending) are left out
