"""Tests for the v0 training script on synthetic data."""

import unittest

import numpy as np

from datara_ml.export_dev_model import synthetic_hourly
from datara_ml.features import build_training_set
from datara_ml.train_v0 import baselines, cross_validate, device_folds, metrics


class TrainV0Test(unittest.TestCase):

    def test_folds_hold_out_whole_devices_once(self):
        devices = np.array([f"d{i % 11}" for i in range(200)])
        folds = device_folds(devices)
        flat = [d for f in folds for d in f]
        self.assertEqual(sorted(flat), sorted(set(devices)))
        self.assertEqual(len(flat), len(set(flat)))

    def test_metrics_of_a_perfect_prediction(self):
        y = np.log1p(np.array([0.0, 1e6, 5e6]))
        m = metrics(y, y)
        self.assertEqual(m["mae_mb"], 0.0)
        self.assertEqual(m["r2_log"], 1.0)

    def test_cross_validation_reports_model_and_baselines(self):
        train = build_training_set(synthetic_hourly(devices=6, hours=24 * 6))
        report = cross_validate(train)
        self.assertEqual(set(report), {"model", "last_24h_avg", "last_hour"})
        self.assertLess(report["model"]["mae_log"], report["last_24h_avg"]["mae_log"])

    def test_baselines_never_missing(self):
        train = build_training_set(synthetic_hourly(devices=2, hours=24 * 3))
        for pred in baselines(train).values():
            self.assertFalse(np.isnan(pred).any())


if __name__ == "__main__":
    unittest.main()
