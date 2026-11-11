# B4/E3 evaluation

1711 hours from 19 devices; every number is on devices held out of training (5-fold grouped by device (train_v0.device_folds, seed 0); Tweedie tuned inside each training fold (4 inner device folds)).

## Next-hour usage error

| Method | MAE (MB) | RMSE (MB) | R² | R² (log) | Predicted ÷ actual total |
|---|---|---|---|---|---|
| Global Tweedie model | 48.7 | 117.3 | 0.26 | -0.34 | 0.90 |
| Last hour | 52.7 | 145.5 | -0.14 | 0.75 | 1.00 |
| Average of last 24 h | 60.2 | 128.3 | 0.11 | 0.06 | 1.07 |
| Same hour yesterday | 73.0 | 171.4 | -0.59 | 0.09 | 1.15 |

1711 held-out hours.

## PET error on simulated quota cycles

Error = projected − actual PET in hours; positive = late. Cycles start at every forecast hour of a held-out device; see “Rules” below.

| Quota | Method | Cycles | Devices | MAE (h) | Median abs (h) | Within ±3 h | Within ±24 h | Bias (h) | Late | Beyond 30 d |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 MB | Global Tweedie model | 1062 | 14 | 6.7 | 1.3 | 67% | 90% | -6.1 | 37% | 0 |
| 100 MB | Last hour | 1062 | 14 | 243.6 | 5.4 | 44% | 61% | 241.3 | 71% | 358 |
| 100 MB | Average of last 24 h | 1062 | 14 | 79.3 | 2.0 | 58% | 81% | 71.6 | 52% | 110 |
| 100 MB | Same hour yesterday | 1062 | 14 | 79.7 | 2.4 | 54% | 82% | 72.2 | 55% | 110 |
| 500 MB | Global Tweedie model | 786 | 14 | 12.3 | 6.0 | 34% | 82% | -8.4 | 42% | 0 |
| 500 MB | Last hour | 786 | 14 | 293.7 | 36.6 | 23% | 46% | 288.4 | 76% | 314 |
| 500 MB | Average of last 24 h | 786 | 14 | 110.1 | 11.5 | 25% | 63% | 98.3 | 59% | 94 |
| 500 MB | Same hour yesterday | 786 | 14 | 110.5 | 13.4 | 25% | 63% | 97.8 | 59% | 94 |
| 1 GB | Global Tweedie model | 500 | 14 | 15.7 | 10.2 | 23% | 78% | -8.3 | 46% | 0 |
| 1 GB | Last hour | 500 | 14 | 280.5 | 41.1 | 14% | 42% | 271.7 | 72% | 182 |
| 1 GB | Average of last 24 h | 500 | 14 | 58.4 | 13.7 | 17% | 62% | 43.5 | 58% | 23 |
| 1 GB | Same hour yesterday | 500 | 14 | 58.5 | 14.7 | 17% | 62% | 43.0 | 58% | 23 |
| 3 GB | Global Tweedie model | 247 | 9 | 40.0 | 30.0 | 6% | 36% | -18.6 | 51% | 0 |
| 3 GB | Last hour | 247 | 9 | 333.3 | 248.0 | 3% | 22% | 313.0 | 73% | 114 |
| 3 GB | Average of last 24 h | 247 | 9 | 101.0 | 29.5 | 8% | 45% | 74.3 | 59% | 16 |
| 3 GB | Same hour yesterday | 247 | 9 | 101.0 | 29.8 | 9% | 42% | 73.9 | 57% | 16 |
| All | Global Tweedie model | 2595 | 15 | 13.3 | 4.6 | 43% | 80% | -8.4 | 42% | 0 |
| All | Last hour | 2595 | 15 | 274.4 | 27.6 | 28% | 49% | 268.2 | 73% | 968 |
| All | Average of last 24 h | 2595 | 15 | 86.7 | 7.5 | 35% | 69% | 74.5 | 56% | 243 |
| All | Same hour yesterday | 2595 | 15 | 87.0 | 8.5 | 34% | 69% | 74.5 | 57% | 243 |

Device-averaged MAE (each device weighs the same, since heavy users give more cycles):

| Quota | Global Tweedie model | Last hour | Average of last 24 h | Same hour yesterday |
|---|---|---|---|---|
| 100 MB | 3.7 | 131.0 | 40.9 | 41.2 |
| 500 MB | 7.5 | 155.5 | 55.7 | 56.0 |
| 1 GB | 11.8 | 150.6 | 65.4 | 65.4 |
| 3 GB | 29.3 | 146.4 | 50.6 | 51.1 |
| All | 10.0 | 178.8 | 87.8 | 88.1 |

### Sensitivity: the final model's hyperparameters in every fold

Nested CV lets each fold pick its own hyperparameters. The shipped model uses `{"tweedie_variance_power": 1.5, "max_depth": 2, "n_estimators": 150}` (tuned on every device); the same held-out evaluation with those fixed in every fold gives next-hour MAE 48.4 MB and:

| Quota | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) | Beyond 30 d |
|---|---|---|---|---|---|---|
| 100 MB | 1062 | 6.7 | 1.3 | 90% | -6.1 | 0 |
| 500 MB | 786 | 11.5 | 5.0 | 82% | -8.7 | 0 |
| 1 GB | 500 | 14.4 | 8.6 | 81% | -8.9 | 0 |
| 3 GB | 247 | 34.8 | 24.2 | 50% | -21.9 | 0 |
| All | 2595 | 12.3 | 4.1 | 82% | -8.9 | 0 |

### Which cycles could be scored

| Quota | Start hours | Scored | Crossing in an unobserved hour | Unknown hours before crossing | Never crossed in the data |
|---|---|---|---|---|---|
| 100 MB | 1711 | 1062 | 133 | 352 | 164 |
| 500 MB | 1711 | 786 | 156 | 574 | 195 |
| 1 GB | 1711 | 500 | 120 | 792 | 299 |
| 3 GB | 1711 | 247 | 128 | 1000 | 336 |

## E3: Global vs Local

Devices that learned ≥ 168 hours (the switch's minimum): **3** of 19; local layer ready (≥ 24 learned hours) on 13; switched to Local on **1** (20 hours shown Local).

| Next hour | Hours | MAE (MB) | RMSE (MB) | Predicted ÷ actual total |
|---|---|---|---|---|
| Global Tweedie model | 1711 | 48.7 | 117.3 | 0.90 |
| Local layer (as switched) | 1711 | 48.8 | 117.4 | 0.90 |
| Global, same rows/cycles | 1302 | 41.6 | 91.3 | 0.92 |
| Local layer (shadow, when ready) | 1302 | 45.1 | 96.3 | 1.04 |

| PET, all quotas | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) |
|---|---|---|---|---|---|
| Global Tweedie model | 2595 | 13.3 | 4.6 | 80% | -8.4 |
| Local layer (as switched) | 2595 | 13.2 | 4.3 | 80% | -8.5 |
| Global, same rows/cycles | 1972 | 14.3 | 5.7 | 79% | -8.7 |
| Local layer (shadow, when ready) | 1972 | 14.5 | 6.8 | 80% | -10.4 |

## Forecast alerts as classification

The app warns when the PET is within 24 h or within 3 h (D8). At every forecast hour and quota of a held-out device, each method's projection is a yes/no prediction, scored against whether measured usage really reached the quota within that time. Positive = reached within the horizon. Precision: when it warned, was it right? Recall: of the times the quota was about to be reached, how often did it warn?

### Within 24h (all quotas: 4343 labelled starts, 2501 not labelable)

Confusion matrix, Global model:

| | Actually reached | Not reached |
|---|---|---|
| **Alert** | 2039 (TP) | 1328 (FP) |
| **No alert** | 124 (FN) | 852 (TN) |

| Quota | Method | Starts | Positives | Precision | Recall | F1 | Accuracy | Specificity |
|---|---|---|---|---|---|---|---|---|
| 100 MB | Global Tweedie model | 1352 | 76% | 0.76 | 1.00 | 0.87 | 0.76 | 0.00 |
| 100 MB | Last hour | 1352 | 76% | 0.97 | 0.66 | 0.79 | 0.72 | 0.93 |
| 100 MB | Average of last 24 h | 1352 | 76% | 0.91 | 0.91 | 0.91 | 0.86 | 0.69 |
| 100 MB | Same hour yesterday | 1352 | 76% | 0.91 | 0.91 | 0.91 | 0.86 | 0.69 |
| 500 MB | Global Tweedie model | 1135 | 61% | 0.60 | 0.97 | 0.74 | 0.59 | 0.00 |
| 500 MB | Last hour | 1135 | 61% | 0.88 | 0.55 | 0.67 | 0.68 | 0.89 |
| 500 MB | Average of last 24 h | 1135 | 61% | 0.77 | 0.77 | 0.77 | 0.72 | 0.64 |
| 500 MB | Same hour yesterday | 1135 | 61% | 0.77 | 0.77 | 0.77 | 0.72 | 0.64 |
| 1 GB | Global Tweedie model | 967 | 41% | 0.38 | 0.76 | 0.50 | 0.38 | 0.12 |
| 1 GB | Last hour | 967 | 41% | 0.74 | 0.54 | 0.62 | 0.73 | 0.87 |
| 1 GB | Average of last 24 h | 967 | 41% | 0.61 | 0.67 | 0.64 | 0.69 | 0.71 |
| 1 GB | Same hour yesterday | 967 | 41% | 0.61 | 0.67 | 0.64 | 0.69 | 0.71 |
| 3 GB | Global Tweedie model | 889 | 5% | 0.36 | 0.84 | 0.51 | 0.92 | 0.92 |
| 3 GB | Last hour | 889 | 5% | 0.24 | 0.59 | 0.34 | 0.89 | 0.90 |
| 3 GB | Average of last 24 h | 889 | 5% | 0.37 | 0.75 | 0.50 | 0.92 | 0.93 |
| 3 GB | Same hour yesterday | 889 | 5% | 0.37 | 0.75 | 0.50 | 0.92 | 0.93 |
| All | Global Tweedie model | 4343 | 50% | 0.61 | 0.94 | 0.74 | 0.67 | 0.39 |
| All | Last hour | 4343 | 50% | 0.85 | 0.60 | 0.70 | 0.75 | 0.89 |
| All | Average of last 24 h | 4343 | 50% | 0.79 | 0.82 | 0.80 | 0.80 | 0.78 |
| All | Same hour yesterday | 4343 | 50% | 0.79 | 0.82 | 0.80 | 0.80 | 0.78 |

### Within 3h (all quotas: 6164 labelled starts, 680 not labelable)

Confusion matrix, Global model:

| | Actually reached | Not reached |
|---|---|---|
| **Alert** | 570 (TP) | 510 (FP) |
| **No alert** | 172 (FN) | 4912 (TN) |

| Quota | Method | Starts | Positives | Precision | Recall | F1 | Accuracy | Specificity |
|---|---|---|---|---|---|---|---|---|
| 100 MB | Global Tweedie model | 1564 | 36% | 0.54 | 0.91 | 0.68 | 0.68 | 0.56 |
| 100 MB | Last hour | 1564 | 36% | 0.80 | 0.72 | 0.76 | 0.83 | 0.90 |
| 100 MB | Average of last 24 h | 1564 | 36% | 0.60 | 0.80 | 0.69 | 0.73 | 0.70 |
| 100 MB | Same hour yesterday | 1564 | 36% | 0.61 | 0.68 | 0.64 | 0.72 | 0.75 |
| 500 MB | Global Tweedie model | 1522 | 9% | 0.44 | 0.38 | 0.41 | 0.90 | 0.95 |
| 500 MB | Last hour | 1522 | 9% | 0.40 | 0.46 | 0.43 | 0.88 | 0.93 |
| 500 MB | Average of last 24 h | 1522 | 9% | 0.51 | 0.53 | 0.52 | 0.91 | 0.95 |
| 500 MB | Same hour yesterday | 1522 | 9% | 0.38 | 0.46 | 0.41 | 0.88 | 0.92 |
| 1 GB | Global Tweedie model | 1531 | 2% | 0.00 | 0.00 | – | 0.98 | 1.00 |
| 1 GB | Last hour | 1531 | 2% | 0.07 | 0.11 | 0.08 | 0.96 | 0.97 |
| 1 GB | Average of last 24 h | 1531 | 2% | 0.03 | 0.04 | 0.03 | 0.96 | 0.98 |
| 1 GB | Same hour yesterday | 1531 | 2% | 0.08 | 0.11 | 0.09 | 0.96 | 0.98 |
| 3 GB | Global Tweedie model | 1547 | 0% | – | – | – | 1.00 | 1.00 |
| 3 GB | Last hour | 1547 | 0% | 0.00 | – | – | 1.00 | 1.00 |
| 3 GB | Average of last 24 h | 1547 | 0% | – | – | – | 1.00 | 1.00 |
| 3 GB | Same hour yesterday | 1547 | 0% | 0.00 | – | – | 1.00 | 1.00 |
| All | Global Tweedie model | 6164 | 12% | 0.53 | 0.77 | 0.63 | 0.89 | 0.91 |
| All | Last hour | 6164 | 12% | 0.66 | 0.65 | 0.65 | 0.92 | 0.95 |
| All | Average of last 24 h | 6164 | 12% | 0.57 | 0.72 | 0.63 | 0.90 | 0.92 |
| All | Same hour yesterday | 6164 | 12% | 0.54 | 0.61 | 0.57 | 0.89 | 0.93 |


## Feature importance (gain, final model)

| Feature | Gain share |
|---|---|
| `log_bytes_last_1h` | 42% |
| `log_avg_bytes_last_3h` | 20% |
| `log_avg_bytes_last_24h` | 9% |
| `log_bytes_same_hour_last_week` | 7% |
| `hour_cos` | 5% |
| `cellular_share_last_hour` | 4% |
| `log_bytes_same_hour_yesterday` | 4% |
| `signal_dbm_last_hour` | 4% |
| `hour_sin` | 4% |
| `day_of_week` | 3% |
| `is_weekend` | 0% |
| `radio_gen_last_hour` | 0% |

## Per device (anonymous)

Counts are the app's hourly runs that made a forecast (including hours never measured afterwards); “learned” counts the measured hours the local layer learned from.

| Device | Fold | Forecast runs | Hours learned | Runs with local ready | Runs shown Local |
|---|---|---|---|---|---|
| D01 | 4 | 318 | 280 | 278 | 0 |
| D02 | 3 | 221 | 149 | 159 | 0 |
| D03 | 5 | 310 | 214 | 263 | 0 |
| D04 | 3 | 270 | 188 | 238 | 30 |
| D05 | 3 | 186 | 163 | 159 | 0 |
| D06 | 3 | 58 | 15 | 0 | 0 |
| D07 | 5 | 58 | 31 | 8 | 0 |
| D08 | 2 | 234 | 99 | 162 | 0 |
| D09 | 1 | 176 | 62 | 99 | 0 |
| D10 | 2 | 78 | 18 | 0 | 0 |
| D11 | 1 | 67 | 16 | 0 | 0 |
| D12 | 1 | 154 | 151 | 130 | 0 |
| D13 | 4 | 105 | 32 | 9 | 0 |
| D14 | 5 | 67 | 16 | 0 | 0 |
| D15 | 4 | 43 | 19 | 0 | 0 |
| D16 | 1 | 52 | 13 | 0 | 0 |
| D17 | 2 | 154 | 89 | 106 | 0 |
| D18 | 2 | 123 | 62 | 77 | 0 |
| D19 | 4 | 104 | 94 | 80 | 0 |

## Rules

- **start_hours**: every training row of a held-out device (observed hour, >= 6 observed of the last 24)
- **quotas_mb**: [100, 500, 1000, 3000]
- **actual_pet**: measured cumulative from the start hour; long NSM windows add their spread total; scored only if the crossing falls in an observed hour
- **error_sign**: projected - actual in hours; positive = late
- **beyond_horizon**: a projection that never reaches the quota in 720 h is scored as start + 720 h
- **alerts**: every start hour and quota; predicted = the projection reaches the quota within W h (AlertRules: 24 h, 3 h); actual = measured usage reached it within W h; starts the data cannot label (a long NSM window across the boundary, a hole, or the data ending) are left out
