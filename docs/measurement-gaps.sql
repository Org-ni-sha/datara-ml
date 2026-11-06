-- Queries behind docs/measurement-gaps.md. Run them on the harvester's cloud database
-- (SQLite Cloud, Dataharvester_v2). Each prints aggregates only, and device ids are cut to 8 characters.

-- Q1. Share of tracked TIME (not rows) inside GAP / INTERRUPTED intervals, per schema version.
--     One GAP row can span minutes to hours, so the row share looks small while the time share is large.
SELECT schema_version,
       COUNT(*) AS n_rows,
       ROUND(100.0 * SUM(measurement_status IN ('GAP', 'INTERRUPTED')) / COUNT(*), 1) AS pct_rows_in_gaps,
       ROUND(100.0 * SUM(CASE WHEN measurement_status IN ('GAP', 'INTERRUPTED')
                              THEN interval_end - interval_start ELSE 0 END)
             / SUM(interval_end - interval_start), 1) AS pct_time_in_gaps
FROM usage_records
WHERE interval_start > 0 AND interval_end > interval_start
GROUP BY schema_version
ORDER BY schema_version;

-- Q2. 30-second cellular bytes by status (MB). GAP rows carry bytes, mostly the phantom jump.
SELECT measurement_status,
       COUNT(*) AS n_rows,
       ROUND(SUM(bytes_total) / 1e6, 1) AS mb
FROM usage_records
WHERE network_type NOT IN ('WIFI', 'NONE')
GROUP BY measurement_status
ORDER BY mb DESC;

-- Q3. Android's own (NetworkStatsManager) mobile total over all devices, every uid incl. hotspot (MB).
SELECT ROUND(SUM(bytes_total) / 1e6, 1) AS nsm_mobile_mb
FROM app_usage_records
WHERE network_type = 'MOBILE';

-- Per-device capture over the SAME hours (30-second vs NSM) needs the hourly rules, so it is
-- computed in Python: python -m datara_ml.capture_report --usage ... --apps ...
