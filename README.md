# datara-ml

The ML pipeline for DATAra (ROADMAP track B): cleaning the data-harvester dataset, training the Global XGBoost next-hour model, and exporting it to ONNX.

Only the B1 phantom-spike repair exists so far.

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
- `hourly_mobile.csv`: one row per device and hour. `mobile_bytes` is the cleaned total and is empty for dropped hours. Use it as the input for the B2 next-hour target.
- `phantom_spike_audit.csv`: every changed or dropped hour, with the 30-second and NSM totals in MB. Keep it as evidence for Chapter 3 and the defense.

### Reading the summary

- "replaced … without (smaller spikes)" counts hours replaced even though no reading was over 50 MB. If that number is large, the 30-second readings and NSM disagree for some other reason (for example an app NSM can't attribute). Check a few of those hours before training.
- Hour buckets use epoch hours. The Philippine offset (UTC+8) is a whole number of hours, so these are also local clock hours.

### Suggested Chapter 3 wording

> Readings recorded immediately after mobile data reconnected could contain the device's lifetime mobile counter, a known artifact of harvester versions ≤ 1.6.7. For each affected device-hour where the 30-second total exceeded 1.5 times the NetworkStatsManager total plus 5 MB, the NetworkStatsManager value was used; hours with no NetworkStatsManager record were excluded. *N* hours across *D* devices were corrected and *M* were excluded.

Fill in *N*, *D* and *M* from the printed summary.
