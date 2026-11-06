"""Tests for the training script on synthetic data."""

import unittest

import numpy as np
import pandas as pd

import datara_ml.train_v0 as tv
from datara_ml.export_dev_model import synthetic_hourly
from datara_ml.features import FEATURES, build_training_set


class TrainV0Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.grid = tv.TWEEDIE_GRID
        tv.TWEEDIE_GRID = {"tweedie_variance_power": [1.3, 1.6], "max_depth": [2], "n_estimators": [60]}
        cls.train = build_training_set(synthetic_hourly(devices=6, hours=24 * 6))
        cls.train[FEATURES] = cls.train[FEATURES].astype(np.float32)

    @classmethod
    def tearDownClass(cls):
        tv.TWEEDIE_GRID = cls.grid

    def test_folds_hold_out_whole_devices_once(self):
        devices = np.array([f"d{i % 11}" for i in range(200)])
        flat = [d for f in tv.device_folds(devices) for d in f]
        self.assertEqual(sorted(flat), sorted(set(devices)))
        self.assertEqual(len(flat), len(set(flat)))

    def test_metrics_of_a_perfect_prediction(self):
        rows = pd.DataFrame({"device_id": ["a"] * 3, "hour_start_ms": [0, 3_600_000, 7_200_000]})
        y = np.array([0.0, 1e6, 5e6])
        m = tv.metrics(rows, y, y)
        self.assertEqual(m["mae_mb"], 0.0)
        self.assertEqual(m["sum_pred_over_actual"], 1.0)
        self.assertEqual(m["median_day_total_error"], 0.0)

    def test_tweedie_predicts_the_mean_and_log1p_smear_is_rescaled(self):
        report = tv.cross_validate(self.train, "tweedie")
        self.assertEqual(set(report), {"model", "last_hour", "last_24h_avg", "params_per_fold"})
        self.assertGreater(report["model"]["sum_pred_over_actual"], 0.7)
        x, y = tv._xy(self.train)
        smear = tv.make("log1p_smear").fit(x, y)
        self.assertGreater(smear.output_scale, 1.0)  # squared error on log1p underpredicts the mean
        self.assertAlmostEqual(smear.predict_bytes(x).sum() / y.sum(), 1.0, places=6)

    def test_tuning_picks_a_grid_point(self):
        self.assertIn(tv.tune(self.train, "tweedie"), tv.grid())
        self.assertEqual(tv.tune(self.train, "log1p_smear"), {})

    def test_baselines_are_bytes_and_never_missing(self):
        for pred in tv.baselines(self.train).values():
            self.assertFalse(np.isnan(pred).any())
            self.assertGreaterEqual(pred.min(), 0)


if __name__ == "__main__":
    unittest.main()
