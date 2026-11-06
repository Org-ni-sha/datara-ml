# B4/E3 evaluation

689 hours from 17 devices; every number is on devices held out of training (5-fold grouped by device (train_v0.device_folds, seed 0); Tweedie tuned inside each training fold (4 inner device folds)).

## Next-hour usage error

| Method | MAE (MB) | RMSE (MB) | R² | R² (log) | Predicted ÷ actual total |
|---|---|---|---|---|---|
| Global Tweedie model | 59.8 | 113.2 | 0.22 | -0.06 | 0.79 |
| Last hour | 62.3 | 124.8 | 0.05 | 0.68 | 0.99 |
| Average of last 24 h | 65.8 | 113.9 | 0.21 | -0.05 | 1.04 |
| Same hour yesterday | 76.3 | 137.4 | -0.15 | 0.01 | 1.07 |

689 held-out hours.

## PET error on simulated quota cycles

Error = projected − actual PET in hours; positive = late. Cycles start at every forecast hour of a held-out device; see “Rules” below.

| Quota | Method | Cycles | Devices | MAE (h) | Median abs (h) | Within ±3 h | Within ±24 h | Bias (h) | Late | Beyond 30 d |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 MB | Global Tweedie model | 489 | 13 | 5.0 | 1.0 | 72% | 98% | -1.2 | 44% | 1 |
| 100 MB | Last hour | 489 | 13 | 177.9 | 2.3 | 52% | 71% | 176.0 | 65% | 118 |
| 100 MB | Average of last 24 h | 489 | 13 | 18.1 | 1.1 | 70% | 95% | 13.7 | 49% | 9 |
| 100 MB | Same hour yesterday | 489 | 13 | 18.7 | 1.2 | 65% | 96% | 14.5 | 50% | 9 |
| 500 MB | Global Tweedie model | 394 | 12 | 26.2 | 5.3 | 36% | 82% | 9.5 | 46% | 9 |
| 500 MB | Last hour | 394 | 12 | 241.3 | 20.2 | 28% | 52% | 235.7 | 72% | 125 |
| 500 MB | Average of last 24 h | 394 | 12 | 39.5 | 9.3 | 30% | 74% | 28.7 | 54% | 12 |
| 500 MB | Same hour yesterday | 394 | 12 | 40.1 | 10.0 | 25% | 72% | 28.4 | 54% | 12 |
| 1 GB | Global Tweedie model | 329 | 9 | 37.3 | 10.0 | 19% | 67% | 5.9 | 39% | 9 |
| 1 GB | Last hour | 329 | 9 | 302.0 | 68.2 | 14% | 36% | 293.0 | 76% | 129 |
| 1 GB | Average of last 24 h | 329 | 9 | 60.7 | 19.0 | 18% | 54% | 43.6 | 59% | 14 |
| 1 GB | Same hour yesterday | 329 | 9 | 60.9 | 18.6 | 17% | 54% | 43.0 | 60% | 14 |
| 3 GB | Global Tweedie model | 155 | 8 | 89.0 | 63.2 | 3% | 33% | -4.3 | 31% | 9 |
| 3 GB | Last hour | 155 | 8 | 357.5 | 569.0 | 3% | 17% | 336.6 | 73% | 78 |
| 3 GB | Average of last 24 h | 155 | 8 | 129.2 | 43.5 | 6% | 34% | 100.2 | 62% | 15 |
| 3 GB | Same hour yesterday | 155 | 8 | 129.5 | 50.6 | 4% | 31% | 99.4 | 63% | 15 |
| All | Global Tweedie model | 1367 | 13 | 28.4 | 4.8 | 41% | 79% | 3.2 | 42% | 28 |
| All | Last hour | 1367 | 13 | 246.4 | 22.7 | 30% | 51% | 239.6 | 71% | 450 |
| All | Average of last 24 h | 1367 | 13 | 47.1 | 6.7 | 39% | 72% | 35.0 | 55% | 50 |
| All | Same hour yesterday | 1367 | 13 | 47.6 | 7.4 | 35% | 71% | 35.0 | 55% | 50 |

Device-averaged MAE (each device weighs the same, since heavy users give more cycles):

| Quota | Global Tweedie model | Last hour | Average of last 24 h | Same hour yesterday |
|---|---|---|---|---|
| 100 MB | 2.5 | 91.3 | 8.7 | 9.0 |
| 500 MB | 15.0 | 138.0 | 41.6 | 41.9 |
| 1 GB | 20.4 | 170.6 | 25.8 | 26.1 |
| 3 GB | 46.2 | 163.2 | 62.6 | 63.7 |
| All | 12.9 | 121.0 | 22.5 | 23.0 |

### Sensitivity: the final model's hyperparameters in every fold

Nested CV lets each fold pick its own hyperparameters. The shipped model uses `{"tweedie_variance_power": 1.9, "max_depth": 2, "n_estimators": 150}` (tuned on every device); the same held-out evaluation with those fixed in every fold gives next-hour MAE 56.2 MB and:

| Quota | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) | Beyond 30 d |
|---|---|---|---|---|---|---|
| 100 MB | 489 | 56.2 | 1.1 | 91% | 52.6 | 37 |
| 500 MB | 394 | 98.7 | 5.8 | 78% | 87.3 | 51 |
| 1 GB | 329 | 120.4 | 13.9 | 66% | 102.6 | 52 |
| 3 GB | 155 | 207.8 | 53.8 | 32% | 166.0 | 46 |
| All | 1367 | 101.1 | 5.0 | 74% | 87.5 | 186 |

### Which cycles could be scored

| Quota | Start hours | Scored | Crossing in an unobserved hour | Unknown hours before crossing | Never crossed in the data |
|---|---|---|---|---|---|
| 100 MB | 689 | 489 | 90 | 53 | 57 |
| 500 MB | 689 | 394 | 92 | 126 | 77 |
| 1 GB | 689 | 329 | 83 | 178 | 99 |
| 3 GB | 689 | 155 | 66 | 234 | 234 |

## E3: Global vs Local

Devices that learned ≥ 168 hours (the switch's minimum): **0** of 17; local layer ready (≥ 24 learned hours) on 9; switched to Local on **0** (0 hours shown Local).

| Next hour | Hours | MAE (MB) | RMSE (MB) | Predicted ÷ actual total |
|---|---|---|---|---|
| Global Tweedie model | 689 | 59.8 | 113.2 | 0.79 |
| Local layer (as switched) | 689 | 59.8 | 113.2 | 0.79 |
| Global, same rows/cycles | 405 | 51.8 | 89.6 | 0.87 |
| Local layer (shadow, when ready) | 405 | 59.0 | 103.5 | 1.07 |

| PET, all quotas | Cycles | MAE (h) | Median abs (h) | Within ±24 h | Bias (h) |
|---|---|---|---|---|---|
| Global Tweedie model | 1367 | 28.4 | 4.8 | 79% | 3.2 |
| Local layer (as switched) | 1367 | 28.4 | 4.8 | 79% | 3.2 |
| Global, same rows/cycles | 856 | 38.6 | 6.3 | 75% | 9.3 |
| Local layer (shadow, when ready) | 856 | 39.3 | 8.8 | 75% | 10.0 |

## Feature importance (gain, final model)

| Feature | Gain share |
|---|---|
| `log_bytes_last_1h` | 23% |
| `log_avg_bytes_last_3h` | 17% |
| `log_avg_bytes_last_24h` | 9% |
| `signal_dbm_last_hour` | 9% |
| `hour_cos` | 8% |
| `radio_gen_last_hour` | 8% |
| `cellular_share_last_hour` | 8% |
| `day_of_week` | 6% |
| `hour_sin` | 4% |
| `is_weekend` | 4% |
| `log_bytes_same_hour_yesterday` | 3% |
| `log_bytes_same_hour_last_week` | 2% |

## Per device (anonymous)

Counts are the app's hourly runs that made a forecast (including hours never measured afterwards); “learned” counts the measured hours the local layer learned from.

| Device | Fold | Forecast runs | Hours learned | Runs with local ready | Runs shown Local |
|---|---|---|---|---|---|
| D01 | 4 | 192 | 156 | 152 | 0 |
| D02 | 4 | 96 | 35 | 34 | 0 |
| D03 | 5 | 185 | 99 | 138 | 0 |
| D04 | 3 | 144 | 98 | 112 | 0 |
| D05 | 2 | 59 | 46 | 32 | 0 |
| D06 | 1 | 28 | 8 | 0 | 0 |
| D07 | 2 | 58 | 31 | 8 | 0 |
| D08 | 3 | 148 | 62 | 76 | 0 |
| D09 | 2 | 152 | 54 | 75 | 0 |
| D10 | 5 | 63 | 17 | 0 | 0 |
| D11 | 4 | 22 | 3 | 0 | 0 |
| D12 | 5 | 41 | 40 | 17 | 0 |
| D13 | 1 | 46 | 12 | 0 | 0 |
| D14 | 3 | 48 | 16 | 0 | 0 |
| D15 | 2 | 3 | 2 | 0 | 0 |
| D16 | 1 | 3 | 2 | 0 | 0 |
| D17 | 1 | 29 | 8 | 0 | 0 |

## Rules

- **start_hours**: every training row of a held-out device (observed hour, >= 6 observed of the last 24)
- **quotas_mb**: [100, 500, 1000, 3000]
- **actual_pet**: measured cumulative from the start hour; long NSM windows add their spread total; scored only if the crossing falls in an observed hour
- **error_sign**: projected - actual in hours; positive = late
- **beyond_horizon**: a projection that never reaches the quota in 720 h is scored as start + 720 h
