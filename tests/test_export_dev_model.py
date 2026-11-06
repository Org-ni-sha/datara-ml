"""Tests for the placeholder model export and the golden parity fixture."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datara_ml.export_dev_model import (
    DEV_MODEL_VERSION, OnnxPredictor, check_parity, export_onnx, fixture_history, golden_cases,
    main, model_version_of, synthetic_hourly,
    train_dev_model, write_fixture,
)
from datara_ml.features import (
    CONTRACT_VERSION, FEATURES, N_FEATURES, PH_OFFSET_MS, HourStats, build_training_set,
    device_histories, feature_vector, project,
)


class ExportTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        hourly = synthetic_hourly(devices=6, hours=24 * 10)
        cls.train = build_training_set(hourly)
        cls.model = train_dev_model(cls.train)
        cls.onnx_path = out / "global_xgboost.onnx"
        export_onnx(cls.model, cls.onnx_path, DEV_MODEL_VERSION)
        cls.predictor = OnnxPredictor(cls.onnx_path)
        history = fixture_history()
        cls.fixture_path = out / "golden_parity.json"
        write_fixture(cls.fixture_path, DEV_MODEL_VERSION, PH_OFFSET_MS, golden_cases(history, cls.predictor))

    @classmethod
    def tearDownClass(cls):
        cls.predictor = None
        cls.tmp.cleanup()

    def test_onnx_signature_and_metadata(self):
        import onnx
        m = onnx.load(str(self.onnx_path))
        self.assertEqual(m.graph.input[0].name, "features")
        dims = m.graph.input[0].type.tensor_type.shape.dim
        self.assertEqual(dims[1].dim_value, N_FEATURES)
        self.assertEqual(m.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        props = {p.key: p.value for p in m.metadata_props}
        self.assertEqual(props["contract_version"], str(CONTRACT_VERSION))
        self.assertEqual(props["model_version"], DEV_MODEL_VERSION)
        self.assertEqual(props["feature_names"].split(","), FEATURES)

    def test_output_transform_metadata(self):
        self.assertEqual(self.predictor.output_transform, "identity")  # Tweedie: output in bytes
        self.assertEqual(self.predictor.output_scale, 1.0)
        x = self.train[FEATURES].to_numpy(np.float32)[:20]
        np.testing.assert_allclose(self.predictor.batch(x), self.model.predict(x), rtol=1e-4)

    def test_transform_must_fit_the_objective(self):
        with self.assertRaises(ValueError):
            export_onnx(self.model, Path(self.tmp.name) / "bad.onnx", "x", output_transform="expm1")
        with self.assertRaises(ValueError):
            export_onnx(self.model, Path(self.tmp.name) / "bad.onnx", "x", output_transform="log")

    def test_onnx_matches_xgboost_including_missing_values(self):
        x = self.train[FEATURES].to_numpy(np.float32)
        self.assertTrue(np.isnan(x).any(), "the parity rows must contain NaN")
        check_parity(self.model, self.predictor, x)

    def test_all_missing_row_matches_xgboost(self):
        x = np.full((1, N_FEATURES), np.nan, dtype=np.float32)
        check_parity(self.model, self.predictor, x)

    def test_export_rejects_wrong_feature_count(self):
        from xgboost import XGBRegressor
        wrong = XGBRegressor(n_estimators=2).fit(np.zeros((4, 3), np.float32), np.zeros(4, np.float32))
        with self.assertRaises(ValueError):
            export_onnx(wrong, Path(self.tmp.name) / "wrong.onnx", "x")

    def test_fixture_replays_against_the_reference(self):
        fixture = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["feature_names"], FEATURES)
        self.assertGreaterEqual(len(fixture["cases"]), 4)

        def nan(v):
            return math.nan if v is None else v

        for case in fixture["cases"]:
            history = {
                h["hour_start_ms"]: HourStats(h["mobile_bytes"], h["readings"], h["cellular_readings"],
                                              nan(h["signal_dbm"]), tuple(h["radio"]))
                for h in case["history"]
            }
            x = feature_vector(history, case["forecast_hour_ms"], fixture["utc_offset_ms"])
            np.testing.assert_array_equal(x, np.array([nan(v) for v in case["expected_features"]], np.float32))
            self.assertAlmostEqual(self.predictor.raw(x), case["expected_output"], delta=abs(case["expected_output"]) * 1e-6)
            self.assertAlmostEqual(self.predictor(x), case["expected_bytes"], delta=case["expected_bytes"] * 1e-6 + 1e-6)
            for p in case["projections"]:
                got = project(history, case["forecast_hour_ms"], p["remaining_bytes"], self.predictor)
                self.assertEqual(got.reach_at_ms, p["reach_at_ms"])

    def test_fixture_only_regenerates_for_an_existing_model(self):
        out = Path(self.tmp.name) / "regen"
        self.assertEqual(model_version_of(self.onnx_path), DEV_MODEL_VERSION)
        main(["--fixture-only", "--model", str(self.onnx_path), "--out", str(out)])
        self.assertEqual((out / "golden_parity.json").read_text(encoding="utf-8"),
                         self.fixture_path.read_text(encoding="utf-8"))

    def test_fixture_covers_missing_values(self):
        fixture = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        features = [v for c in fixture["cases"] for v in c["expected_features"]]
        self.assertIn(None, features)
        reaches = [p["reach_at_ms"] for c in fixture["cases"] for p in c["projections"]]
        self.assertIn(None, reaches)


if __name__ == "__main__":
    unittest.main()
