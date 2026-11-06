"""Tests for the NSM hourly rule (feature contract v2)."""

import math
import unittest

import pandas as pd

from datara_ml.nsm_hourly import (
    COVERAGE_TOLERANCE_MS, HOUR_MS, MAX_WINDOW_MS, Window, hourly_from_windows, nsm_hourly,
    windows_from_app_rows,
)

H0 = 1_794_153_600_000  # 2026-11-09 00:00 Asia/Manila (16:00 UTC the day before)
M = 60_000


class SpreadTest(unittest.TestCase):

    def test_bytes_are_spread_evenly_over_the_window(self):
        h = hourly_from_windows([Window(H0 - 30 * M, H0 + 30 * M, 6_000_000), Window(H0 + 30 * M, H0 + 60 * M, 1)])
        self.assertTrue(math.isnan(h[H0 - HOUR_MS]))     # only half covered
        self.assertAlmostEqual(h[H0], 3_000_001)

    def test_two_hour_window_is_usable_and_longer_is_not(self):
        ok = hourly_from_windows([Window(H0, H0 + MAX_WINDOW_MS, 8_000_000, "GAP")])
        self.assertEqual(ok, {H0: 4_000_000, H0 + HOUR_MS: 4_000_000})
        long = hourly_from_windows([Window(H0, H0 + MAX_WINDOW_MS + 1, 8_000_000, "GAP")])
        self.assertTrue(all(math.isnan(v) for v in long.values()))

    def test_long_zero_window_is_a_measured_zero(self):
        h = hourly_from_windows([Window(H0, H0 + 10 * HOUR_MS, 0, "GAP")])
        self.assertEqual(h, {H0 + i * HOUR_MS: 0 for i in range(10)})
        mixed = hourly_from_windows([Window(H0, H0 + 30 * M, 2_000_000), Window(H0 + 30 * M, H0 + 4 * HOUR_MS, 0, "GAP")])
        self.assertEqual(mixed[H0], 2_000_000)
        self.assertEqual(mixed[H0 + 3 * HOUR_MS], 0)

    def test_measured_zero_is_observed_and_reset_is_not(self):
        h = hourly_from_windows([Window(H0, H0 + HOUR_MS, 0), Window(H0 + HOUR_MS, H0 + 2 * HOUR_MS, 0, "COUNTER_RESET")])
        self.assertEqual(h[H0], 0)
        self.assertTrue(math.isnan(h[H0 + HOUR_MS]))

    def test_uncovered_time_beyond_tolerance_and_overlaps_are_unobserved(self):
        gap = COVERAGE_TOLERANCE_MS + 1
        self.assertTrue(math.isnan(hourly_from_windows([Window(H0 + gap, H0 + HOUR_MS, 1)])[H0]))
        self.assertEqual(hourly_from_windows([Window(H0 + COVERAGE_TOLERANCE_MS, H0 + HOUR_MS, 1)])[H0], 1)
        overlap = hourly_from_windows([Window(H0, H0 + 40 * M, 1), Window(H0 + 30 * M, H0 + HOUR_MS, 1)])
        self.assertTrue(math.isnan(overlap[H0]))


class AppRowsTest(unittest.TestCase):

    def rows(self):
        def r(start, end, net, b, pkg="app", status="OK"):
            return {"device_id": "a", "start_time": start, "end_time": end, "network_type": net, "bytes_total": b,
                    "measurement_status": status, "utc_offset_minutes": 480, "package_name": pkg}
        return pd.DataFrame([
            r("2026-11-09T00:00:00", "2026-11-09T00:30:00", "MOBILE", 1_000_000),
            r("2026-11-09T00:00:00", "2026-11-09T00:30:00", "MOBILE", 2_000_000, pkg="hotspot"),
            r("2026-11-09T00:00:00", "2026-11-09T00:30:00", "WIFI", 9_000_000),
            r("2026-11-09T00:30:00", "2026-11-09T01:00:00", "WIFI", 5_000_000),  # Wi-Fi only: measured 0 mobile
        ])

    def test_windows_use_local_time_and_count_every_uid(self):
        w = windows_from_app_rows(self.rows())
        self.assertEqual(list(w["start_ms"]), [H0, H0 + 30 * M])
        self.assertEqual(list(w["mobile_bytes"]), [3_000_000, 0])  # hotspot included, Wi-Fi excluded

    def test_hourly_from_export(self):
        h = nsm_hourly(self.rows()).set_index("hour_start_ms")
        self.assertEqual(h.loc[H0, "mobile_bytes"], 3_000_000)

    def test_a_bad_row_marks_its_window(self):
        rows = self.rows()
        rows.loc[0, "measurement_status"] = "CLOCK_ERROR"
        self.assertEqual(windows_from_app_rows(rows).loc[0, "status"], "CLOCK_ERROR")
        self.assertTrue(math.isnan(nsm_hourly(rows).set_index("hour_start_ms").loc[H0, "mobile_bytes"]))


if __name__ == "__main__":
    unittest.main()
