"""Tests for the B2 feature contract (datara_ml.features)."""

import math
import unittest

import numpy as np
import pandas as pd

from datara_ml.features import (
    FEATURES, HOUR_MS, MIN_OBSERVED_LAST_24H, N_FEATURES, PH_OFFSET_MS, HourStats,
    build_hourly_table, build_training_set, feature_vector, hourly_network, median,
    pooled_network, project, radio_gen_of,
)

# 2026-11-09 00:00 Asia/Manila (a Monday), as epoch ms
MON_MIDNIGHT_PH = 1_794_153_600_000
MB = 1_000_000


def idx(name):
    return FEATURES.index(name)


def stats(mb, readings=120, cellular=120, signal=-90.0, radio=(0, 0, 120, 0)):
    return HourStats(mobile_bytes=mb * MB, readings=readings, cellular_readings=cellular,
                     signal_dbm=signal, radio=radio)


class ContractShapeTest(unittest.TestCase):

    def test_twelve_features_in_contract_order(self):
        self.assertEqual(N_FEATURES, 12)
        self.assertEqual(FEATURES[0], "hour_sin")
        self.assertEqual(FEATURES[11], "cellular_share_last_hour")

    def test_vector_is_float32_with_n_features(self):
        x = feature_vector({}, MON_MIDNIGHT_PH)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.shape, (N_FEATURES,))

    def test_forecast_hour_must_be_an_hour_boundary(self):
        with self.assertRaises(ValueError):
            feature_vector({}, MON_MIDNIGHT_PH + 1)


class TimeFeatureTest(unittest.TestCase):

    def test_monday_midnight_philippine_time(self):
        x = feature_vector({}, MON_MIDNIGHT_PH)
        self.assertAlmostEqual(x[idx("hour_sin")], 0.0, places=6)
        self.assertAlmostEqual(x[idx("hour_cos")], 1.0, places=6)
        self.assertEqual(x[idx("day_of_week")], 0)
        self.assertEqual(x[idx("is_weekend")], 0)

    def test_saturday_18h_is_weekend(self):
        t = MON_MIDNIGHT_PH + (5 * 24 + 18) * HOUR_MS
        x = feature_vector({}, t)
        self.assertEqual(x[idx("day_of_week")], 5)
        self.assertEqual(x[idx("is_weekend")], 1)
        self.assertAlmostEqual(x[idx("hour_sin")], -1.0, places=6)

    def test_utc_offset_changes_local_hour(self):
        x = feature_vector({}, MON_MIDNIGHT_PH, offset_ms=0)  # 16:00 Sunday UTC
        self.assertEqual(x[idx("day_of_week")], 6)
        self.assertAlmostEqual(x[idx("hour_cos")], math.cos(2 * math.pi * 16 / 24), places=6)


class UsageFeatureTest(unittest.TestCase):

    def setUp(self):
        self.t = MON_MIDNIGHT_PH + 200 * HOUR_MS
        self.h = {self.t - j * HOUR_MS: stats(j) for j in range(1, 170)}

    def test_lags_are_log1p_bytes(self):
        x = feature_vector(self.h, self.t)
        self.assertAlmostEqual(x[idx("log_bytes_last_1h")], math.log1p(1 * MB), places=5)
        self.assertAlmostEqual(x[idx("log_bytes_same_hour_yesterday")], math.log1p(24 * MB), places=5)
        self.assertAlmostEqual(x[idx("log_bytes_same_hour_last_week")], math.log1p(168 * MB), places=5)

    def test_averages_are_log1p_of_the_mean_over_observed_hours(self):
        del self.h[self.t - 2 * HOUR_MS]
        x = feature_vector(self.h, self.t)
        self.assertAlmostEqual(x[idx("log_avg_bytes_last_3h")], math.log1p((1 + 3) / 2 * MB), places=5)
        expected24 = (sum(range(1, 25)) - 2) / 23 * MB
        self.assertAlmostEqual(x[idx("log_avg_bytes_last_24h")], math.log1p(expected24), places=5)

    def test_missing_hours_are_nan(self):
        x = feature_vector({}, self.t)
        for name in FEATURES[4:]:
            self.assertTrue(math.isnan(x[idx(name)]), name)

    def test_zero_usage_hour_is_observed_not_missing(self):
        x = feature_vector({self.t - HOUR_MS: stats(0)}, self.t)
        self.assertEqual(x[idx("log_bytes_last_1h")], 0.0)


class NetworkFeatureTest(unittest.TestCase):

    def test_last_hour_network(self):
        t = MON_MIDNIGHT_PH + 10 * HOUR_MS
        x = feature_vector({t - HOUR_MS: stats(1, readings=100, cellular=25, signal=-97.5,
                                                radio=(0, 5, 20, 0))}, t)
        self.assertEqual(x[idx("signal_dbm_last_hour")], -97.5)
        self.assertEqual(x[idx("radio_gen_last_hour")], 4)
        self.assertAlmostEqual(x[idx("cellular_share_last_hour")], 0.25, places=6)

    def test_radio_tie_goes_to_higher_generation_and_none_is_nan(self):
        self.assertEqual(radio_gen_of((0, 3, 3, 0)), 4)
        self.assertEqual(radio_gen_of((0, 0, 3, 3)), 5)
        self.assertTrue(math.isnan(radio_gen_of((0, 0, 0, 0))))

    def test_pooled_network_over_24_hours(self):
        t = MON_MIDNIGHT_PH + 30 * HOUR_MS
        h = {
            t - 1 * HOUR_MS: stats(1, readings=100, cellular=0, signal=math.nan, radio=(0, 0, 0, 0)),
            t - 2 * HOUR_MS: stats(1, readings=100, cellular=100, signal=-80.0, radio=(0, 60, 40, 0)),
            t - 3 * HOUR_MS: stats(1, readings=100, cellular=50, signal=-100.0, radio=(0, 0, 50, 0)),
            t - 25 * HOUR_MS: stats(1, readings=100, cellular=100, signal=-50.0, radio=(100, 0, 0, 0)),
        }
        n = pooled_network(h, t)
        self.assertEqual(n.signal_dbm, -90.0)        # median of -80 and -100; -50 is too old
        self.assertEqual(n.radio_gen, 4)             # LTE 90 vs 3G 60
        self.assertAlmostEqual(n.cellular_share, 150 / 300)

    def test_median_averages_middle_two(self):
        self.assertEqual(median([-90, -80, -70, -100]), -85.0)
        self.assertTrue(math.isnan(median([])))


class HourlyTableTest(unittest.TestCase):

    def test_hourly_network_from_raw_readings(self):
        h0 = MON_MIDNIGHT_PH
        usage = pd.DataFrame([
            {"device_id": "a", "timestamp": h0 + 30_000, "network_type": "LTE", "signal_strength": -91},
            {"device_id": "a", "timestamp": h0 + 60_000, "network_type": "LTE", "signal_strength": -999},
            {"device_id": "a", "timestamp": h0 + 90_000, "network_type": "WIFI", "signal_strength": -85},
            {"device_id": "a", "timestamp": h0 + 120_000, "network_type": "5G", "signal_strength": 0},
            {"device_id": "a", "timestamp": h0 + 150_000, "network_type": "CELLULAR", "signal_strength": -80},
            {"device_id": "a", "timestamp": h0 + HOUR_MS, "network_type": "NONE", "signal_strength": -999},
        ])
        net = hourly_network(usage).set_index("hour_start_ms")
        first = net.loc[h0]
        self.assertEqual(first["readings"], 5)
        self.assertEqual(first["cellular_readings"], 4)          # LTE, LTE, 5G, CELLULAR
        self.assertEqual(first["signal_dbm"], -85.0)             # median of -91, -85, -80
        self.assertEqual((first["radio_2g"], first["radio_3g"], first["radio_lte"], first["radio_5g"]),
                         (0, 0, 2, 1))
        self.assertTrue(math.isnan(net.loc[h0 + HOUR_MS]["signal_dbm"]))

    def test_dropped_hours_are_not_observed(self):
        t = MON_MIDNIGHT_PH + 30 * HOUR_MS
        hourly_mobile = pd.DataFrame([
            {"device_id": "a", "hour_start_ms": t - HOUR_MS, "mobile_bytes": np.nan},
        ])
        table = build_hourly_table(hourly_mobile, hourly_network(pd.DataFrame(
            [{"device_id": "a", "timestamp": t - 10, "network_type": "LTE", "signal_strength": -90}])))
        self.assertEqual(len(table), 1)
        self.assertEqual(len(build_training_set(table, min_observed_last_24h=0)), 0)


class TrainingSetTest(unittest.TestCase):

    def test_rows_need_enough_observed_history_and_target_is_log1p(self):
        rows = [{"device_id": "a", "hour_start_ms": MON_MIDNIGHT_PH + k * HOUR_MS, "mobile_bytes": float(k * MB),
                 "readings": 120, "cellular_readings": 120, "signal_dbm": -90.0,
                 "radio_2g": 0, "radio_3g": 0, "radio_lte": 120, "radio_5g": 0} for k in range(30)]
        train = build_training_set(pd.DataFrame(rows))
        self.assertEqual(train["hour_start_ms"].min(), MON_MIDNIGHT_PH + MIN_OBSERVED_LAST_24H * HOUR_MS)
        self.assertEqual(list(train.columns[2:-1]), FEATURES)
        last = train.iloc[-1]
        self.assertAlmostEqual(last["target"], math.log1p(29 * MB))
        self.assertAlmostEqual(last["log_bytes_last_1h"], math.log1p(28 * MB), places=4)


class ProjectionTest(unittest.TestCase):

    def setUp(self):
        self.t = MON_MIDNIGHT_PH + 48 * HOUR_MS
        self.h = {self.t - j * HOUR_MS: stats(10) for j in range(1, 30)}
        self.ten_mb = lambda x: 10 * MB

    def test_reach_is_interpolated_inside_the_hour(self):
        p = project(self.h, self.t, 25 * MB, self.ten_mb)
        self.assertAlmostEqual(p.next_hour_bytes, 10 * MB, delta=1)
        self.assertAlmostEqual(p.reach_at_ms, self.t + 2.5 * HOUR_MS, delta=2)

    def test_beyond_horizon_is_none(self):
        p = project(self.h, self.t, 10 * MB * 721, self.ten_mb)
        self.assertIsNone(p.reach_at_ms)

    def test_nothing_remaining_reaches_now(self):
        self.assertEqual(project(self.h, self.t, 0, self.ten_mb).reach_at_ms, self.t)

    def test_predictions_feed_later_lags(self):
        seen = []

        def model(x):
            seen.append(float(x[idx("log_bytes_last_1h")]))
            return 2 * MB

        project(self.h, self.t, 5 * MB, model)
        self.assertAlmostEqual(seen[0], math.log1p(10 * MB), places=4)   # real last hour
        self.assertAlmostEqual(seen[1], math.log1p(2 * MB), places=4)    # the first prediction

    def test_zero_predictions_never_reach(self):
        self.assertIsNone(project(self.h, self.t, 1, lambda x: 0.0, horizon_hours=5).reach_at_ms)


if __name__ == "__main__":
    unittest.main()
