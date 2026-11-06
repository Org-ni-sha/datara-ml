# Measurement gaps: why hourly usage comes from NetworkStatsManager

Feature contract v2 (2026-11-06) measures each hour's mobile data with Android's own usage records (NetworkStatsManager, "NSM") instead of the harvester's 30-second readings. This page explains why, with numbers that can be reproduced from the harvester database.

## Two ways the phone measures mobile data

| | 30-second readings (`usage_records`) | NetworkStatsManager (`app_usage_records`) |
|---|---|---|
| What it is | The app reads the phone's mobile-data counter (`TrafficStats`) every 30 seconds and stores the difference, like reading a car's odometer at regular stops. | Android itself records how much data every app used, in its own database. The app asks it every 10 minutes how much was used since the last question. |
| Needs the app awake? | Yes. If the app is paused, no reading is taken. | No. Android keeps counting while the app sleeps; the next question returns everything since the last one. |
| Detail | Fine-grained (30 s), plus signal and network type at each reading. | Per 10-minute window when the app is awake; longer windows after the app slept. |

## Problem 1: the app is put to sleep

Phone makers (MIUI, vivo and others) put background apps into deep sleep to save battery, and the 30-second timer stops. When it wakes, the harvester records one long **GAP** interval covering the whole pause. One GAP row can cover minutes or hours, so counting rows hides the problem; counting time shows it:

| Harvester schema | Version | **Time** in GAP/INTERRUPTED |
|---|---|---|
| 1 | 1.6.3 | **63.5%** |
| 2 | 1.6.5–1.6.6 | **65.6%** |
| 3 | 1.6.7 | **62.1%** |
| 4 | 1.6.8 (heartbeat) | **45.3%** |

By row count only 1.9–5.6% of rows are GAP/INTERRUPTED, which is why the problem was easy to miss.

(Cloud database, 2026-11-06, query Q1. The 1.6.8 heartbeat helps but about half the time is still not covered by regular readings.)

## Problem 2: the phantom counter jump

On many phones the mobile counter only adds up interfaces that are switched on. When mobile data comes back after Wi-Fi or no signal, the counter can jump to the phone's lifetime total, and harvester ≤ 1.6.7 recorded that jump as usage. Version 1.6.8 fixes it, and `phantom_spikes.py` repairs older rows (README, "B1").

Gaps often end exactly when mobile data comes back, so **the bytes on GAP rows are mostly this jump, not real use**. In the 2026-11-06 export:

- 30-second cellular bytes on GAP rows: **255,301 MB**; on OK rows: 216,659 MB (Q2)
- Android's own total mobile use over all devices and the whole period: **133,778 MB** (Q3)

GAP rows alone claim almost twice what Android measured in total, so they cannot be added back to fill the gaps.

## The result: the 30-second readings miss about a quarter of usage

Over the hours that **both** methods measured (so sleep is not counted against the 30-second method), per device (2026-11-06 export, `capture_report.py`):

| Device | Hours | NSM (MB) | 30-second (MB) | Captured |
|---|---|---|---|---|
| ccc789e2 | 10 | 1,523.3 | 4.5 | **0.3%** |
| 365f9caf | 78 | 6,476.4 | 79.5 | **1.2%** |
| f123ea12 | 23 | 5,500.7 | 2,351.2 | **42.7%** |
| ed4c65bc | 2 | 226.9 | 139.3 | **61.4%** |
| acdf17f5 | 30 | 4,126.1 | 2,675.9 | **64.9%** |
| be83fd1d | 11 | 827.7 | 657.7 | **79.5%** |
| 29b00020 | 18 | 2,477.4 | 2,229.8 | 90.0% |
| bd6d6db8 | 64 | 6,309.9 | 5,778.1 | 91.6% |
| ef97f9f3 | 117 | 3,473.2 | 3,227.6 | 92.9% |
| 23d746a8 | 71 | 3,188.9 | 3,019.2 | 94.7% |
| 8bd212be | 40 | 10,389.3 | 9,891.3 | 95.2% |
| a8c3e5d1 | 46 | 3,650.1 | 3,489.3 | 95.6% |
| f05ba8ad | 65 | 15,594.7 | 14,985.5 | 96.1% |
| 883d517d | 23 | 3,823.3 | 3,808.6 | 99.6% |
| 566b8894 | 10 | 987.8 | 988.1 | 100.0% |
| **All** | **608** | **68,576** | **53,326** | **77.8%** |

Most phones agree within about 10%. A few undercount badly, most likely because `TrafficStats` does not count all mobile traffic on those models. The team's own query over the full cloud database on 2026-11-06 (18 devices) found the same pattern, including one more device at 0.1%.

On the lead's POCO F3, DATAra (which reads both) showed the same thing: a 36-hour pause of its 30-second timer was fully covered by NSM (908 MB), and while awake the two methods agreed within 2–6%.

## Why it matters for the forecast

The model learns "how much will this person use next hour" from past hours. If the past hours are undercounted, the model learns lower usage than real, and the projection reaches the quota **later than it really will**: the PET would be systematically late, the worst direction for a warning.

## The fix (feature contract v2)

Hourly mobile bytes come from NSM:

1. Each NSM window's mobile bytes (all apps, including hotspot sharing) are **spread evenly** over the window and added to the clock hours it covers.
2. An hour counts as **measured** only if every window touching it is **at most 2 hours long**, the windows don't overlap, and together they cover the hour (up to 1 minute missing).
3. Hours that fail are **unmeasured**: they are not used as training targets and appear as missing values in the features. A measured zero (the phone was on Wi-Fi) counts.

The app does exactly the same with its own 10-minute NSM readings (`durable_readings`), so training and the phone agree. Signal, radio generation and share of time on mobile data still come from the 30-second readings, because NSM does not record them; they are missing for an hour with no 30-second readings.

## Limits

- **Spreading is an estimate.** Within a window up to 2 hours long, use is assumed even. A burst at the start of a 2-hour window is split across two hours.
- **Lost hours.** Hours inside longer windows (the app slept for more than 2 hours) are dropped. In the 2026-11-06 export, 648 of 2,388 hours touched by NSM windows are measured; 1,623 were lost to long windows and 117 to small coverage holes. This cuts the training rows from 841 (v1) to 476, from 13 devices instead of 17.
- **The 2-hour limit** is a judgement call (`MAX_WINDOW_MS` in `nsm_hourly.py`): longer windows give more hours but blur when the data was used. Android itself stores usage in buckets of about 2 hours.
- `phantom_spikes.py` and its audit stay in the repo as evidence of the counter artifact, but the target no longer depends on them.

## Suggested Chapter 3 paragraph

> Hourly mobile usage was measured from Android's NetworkStatsManager records rather than the 30-second TrafficStats readings. Device vendors suspend background apps, and in the collected data 45–66% of tracked time fell in suspended intervals, during which no 30-second readings were taken; the bytes recorded when those intervals ended were dominated by a known counter artifact and could not be used. Over hours measured by both methods, the 30-second readings captured 77.8% of the usage recorded by NetworkStatsManager, and as little as 0.3% on some devices. Each NetworkStatsManager window's bytes were distributed evenly over the clock hours it covered; an hour was treated as measured only when it was fully covered by non-overlapping windows of at most two hours, and other hours were excluded as targets and treated as missing inputs. The application computes the same hourly values on the device from its own NetworkStatsManager readings.

## Reproducing the numbers

- Q1–Q3: [`measurement-gaps.sql`](measurement-gaps.sql), run on the harvester cloud database (the tables above used the 2026-11-06 data; Q1 from the full database, Q2–Q3 from the export).
- Per-device capture: `python -m datara_ml.capture_report --usage data/usage_records.csv --apps data/app_usage_records.csv`
- Measured/lost hours: `python -m datara_ml.features --apps data/app_usage_records.csv --usage data/usage_records.csv --out data/train.csv` prints them.
