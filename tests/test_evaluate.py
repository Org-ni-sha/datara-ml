"""Tests for the B4/E3 evaluation on synthetic data."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import datara_ml.evaluate as ev
from datara_ml.export_dev_model import synthetic_hourly
from datara_ml.features import HOUR_MS, HourStats, PROJECTION_HORIZON_HOURS, device_histories, project
from datara_ml.nsm_hourly import nsm_hourly

T0 = 1_793_000_000_000 // HOUR_MS * HOUR_MS
PARAMS = {"tweedie_variance_power": 1.5, "max_depth": 2, "n_estimators": 30}


def hist_of(values: dict[int, float]) -> dict[int, HourStats]:
    return {T0 + k * HOUR_MS: HourStats(mobile_bytes=v) for k, v in values.items()}


class CrossingTest(unittest.TestCase):

    def test_crossing_is_interpolated_inside_an_observed_hour(self):
        hist = hist_of({0: 60e6, 1: 0.0, 2: 80e6})
        known = {h: s.mobile_bytes for h, s in hist.items()}
        c = ev.actual_crossing(hist, known, T0, 100e6, T0 + 2 * HOUR_MS)
        self.assertEqual(c.status, "crossed")
        self.assertEqual(c.reach_ms, T0 + 2 * HOUR_MS + HOUR_MS // 2)  # 40 of the hour's 80 MB

    def test_long_windows_carry_their_total_but_cannot_hold_the_crossing(self):
        hist = hist_of({0: 50e6, 4: 100e6})
        known = {h: s.mobile_bytes for h, s in hist.items()}
        known.update({T0 + k * HOUR_MS: 10e6 for k in (1, 2, 3)})  # a 3-hour window of 30 MB
        self.assertEqual(ev.actual_crossing(hist, known, T0, 130e6, T0 + 4 * HOUR_MS),
                         ev.Crossing("crossed", T0 + 4 * HOUR_MS + HOUR_MS // 2))
        self.assertEqual(ev.actual_crossing(hist, known, T0, 70e6, T0 + 4 * HOUR_MS).status,
                         "crossing_unobserved")

    def test_unknown_hours_and_the_end_of_the_data_are_not_scored(self):
        hist = hist_of({0: 10e6, 2: 500e6})
        known = {h: s.mobile_bytes for h, s in hist.items()}
        self.assertEqual(ev.actual_crossing(hist, known, T0, 100e6, T0 + 2 * HOUR_MS).status, "unknown_gap")
        self.assertEqual(ev.actual_crossing(hist, known, T0, 5e9, T0 + 0 * HOUR_MS).status, "never_crossed")

    def test_known_hourly_spreads_long_windows_and_matches_nsm_on_observed_hours(self):
        def iso(ms):
            return pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
        rows = [  # utc_offset 0 so local text = UTC
            (T0, T0 + HOUR_MS, 6e6),                      # observed hour 0
            (T0 + HOUR_MS, T0 + 4 * HOUR_MS, 30e6),       # 3-hour window: unobserved, 10 MB/h known
            (T0 + 4 * HOUR_MS, T0 + 5 * HOUR_MS, 2e6),    # observed hour 4
        ]
        apps = pd.DataFrame([{"device_id": "a", "start_time": iso(s), "end_time": iso(e), "network_type": "MOBILE",
                              "bytes_total": b, "measurement_status": "OK", "utc_offset_minutes": 0}
                             for s, e, b in rows])
        from datara_ml.nsm_hourly import windows_from_app_rows
        known = ev.known_hourly(windows_from_app_rows(apps))["a"]
        self.assertEqual([known[T0 + k * HOUR_MS] for k in range(5)], [6e6, 10e6, 10e6, 10e6, 2e6])
        nsm = nsm_hourly(apps).set_index("hour_start_ms")["mobile_bytes"]
        self.assertTrue(math.isnan(nsm[T0 + 2 * HOUR_MS]))
        self.assertEqual(nsm[T0 + 4 * HOUR_MS], known[T0 + 4 * HOUR_MS])


class ProjectionTest(unittest.TestCase):

    def test_one_trajectory_gives_project_s_pet_for_every_quota(self):
        hist = {h: s for h, s in device_histories(synthetic_hourly(devices=1, hours=24 * 4)).popitem()[1].items()}
        start = max(hist)

        def predict(x):  # depends on the lags, so the trajectory feeds back like the model's
            return 1e6 + 0.5 * math.expm1(x[4]) if not math.isnan(x[4]) else 3e6

        bs = ev.trajectory(hist, start, 5e9, predict)
        for q in (0.0, 1.0, 2.5e6, 40e6, 777e6, 5e9):
            self.assertEqual(ev.reach_from_trajectory(bs, start, q), project(hist, start, q, predict).reach_at_ms, q)

    def test_never_reaching_is_beyond_the_horizon(self):
        bs = ev.trajectory(hist_of({0: 0.0}), T0 + HOUR_MS, 1e6, lambda x: 0.0)
        self.assertEqual(len(bs), PROJECTION_HORIZON_HOURS)
        self.assertIsNone(ev.reach_from_trajectory(bs, T0 + HOUR_MS, 1e6))

    def test_baselines(self):
        x = np.full(12, np.nan)
        x[6] = math.log1p(5e6)
        self.assertAlmostEqual(ev.baseline_bytes("last_hour", x), 5e6, delta=1)   # falls back to 24 h avg
        x[4], x[7] = math.log1p(9e6), math.log1p(2e6)
        self.assertAlmostEqual(ev.baseline_bytes("last_hour", x), 9e6, delta=1)
        self.assertAlmostEqual(ev.baseline_bytes("same_hour_yesterday", x), 2e6, delta=1)
        self.assertEqual(ev.baseline_bytes("last_24h_avg", np.full(12, np.nan)), 0.0)
        rep = ev.repeated_first(lambda v: float(v[0]))
        self.assertEqual([rep(np.array([3.0])), rep(np.array([8.0]))], [3.0, 3.0])


class LocalLayerTest(unittest.TestCase):
    """The port of LocalLayer.kt / ModelSwitch."""

    def test_ratio_is_clamped_and_neutral_without_predictions(self):
        self.assertEqual(ev.ratio(10.0, 1.0), 5.0)
        self.assertEqual(ev.ratio(0.0, 1.0), 0.2)
        self.assertEqual(ev.ratio(3.0, 0.0), 1.0)
        self.assertEqual([ev.block(h) for h in (0, 5, 6, 11, 12, 18, 23)], [0, 0, 1, 1, 2, 3, 3])

    def test_block_ratio_after_12_hours_else_overall(self):
        p = ev.LocalParams(overall_actual=2.0, overall_predicted=1.0, overall_hours=30,
                           block_actual=[3.0, 0, 0, 0], block_predicted=[1.0, 0, 0, 0], block_hours=[11, 0, 0, 0])
        self.assertEqual(p.factor(2), 2.0)
        p.block_hours[0] = 12
        self.assertEqual(p.factor(2), 3.0)

    def test_learn_first_value_then_ewma_and_drops_unobserved(self):
        p = ev.LocalParams(pending_hour=T0, pending_local_hour=7, pending_global_bytes=100.0)
        p1, s1 = ev.learn(p, ev.LayerScores(), 300.0)
        self.assertEqual((p1.overall_actual, p1.overall_predicted, p1.overall_hours), (300.0, 100.0, 1))
        self.assertEqual((p1.block_actual[1], p1.block_hours[1]), (300.0, 1))
        self.assertEqual((s1.hours_seen, s1.global_recent_mae, s1.local_recent_mae), (1, 200.0, None))
        self.assertIsNone(p1.pending_hour)
        p1.pending_hour, p1.pending_local_hour, p1.pending_global_bytes, p1.pending_local_bytes = T0 + HOUR_MS, 8, 100.0, 250.0
        p2, s2 = ev.learn(p1, s1, 200.0)
        self.assertAlmostEqual(p2.overall_actual, 300.0 + 0.05 * (200.0 - 300.0))
        self.assertAlmostEqual(s2.global_recent_mae, 200.0 + (100.0 - 200.0) / 168)
        self.assertEqual(s2.local_recent_mae, 50.0)
        p2.pending_hour, p2.pending_global_bytes = T0 + 2 * HOUR_MS, 10.0
        p3, s3 = ev.learn(p2, s2, None)
        self.assertEqual((p3.overall_hours, s3), (2, s2))
        self.assertEqual(p3.last_learned_hour, T0 + 2 * HOUR_MS)

    def test_switch_needs_168_hours_and_5_percent_and_is_reversible(self):
        s = ev.LayerScores(hours_seen=167, global_recent_mae=100.0, local_recent_mae=10.0)
        self.assertEqual(ev.decide(ev.GLOBAL, s, True), ev.GLOBAL)
        s.hours_seen = 168
        self.assertEqual(ev.decide(ev.GLOBAL, s, False), ev.GLOBAL)
        self.assertEqual(ev.decide(ev.GLOBAL, s, True), ev.LOCAL)
        s.local_recent_mae = 96.0  # better, but not by 5%
        self.assertEqual(ev.decide(ev.GLOBAL, s, True), ev.GLOBAL)
        self.assertEqual(ev.decide(ev.LOCAL, s, True), ev.LOCAL)
        s.local_recent_mae = 100.0
        self.assertEqual(ev.decide(ev.LOCAL, s, True), ev.GLOBAL)

    def test_replay_learns_the_ratio_and_switches_after_a_week(self):
        hours = 24 * 10
        hist = hist_of({k: 20e6 for k in range(hours)})  # the person uses twice what Global predicts
        runs = ev.replay_device(hist, T0, T0 + (hours - 1) * HOUR_MS, lambda x: 10e6)
        self.assertEqual(runs[0].hour_ms, T0 + 6 * HOUR_MS)  # needs 6 observed of the last 24
        self.assertTrue(all(r.active_model == ev.GLOBAL for r in runs if r.hours_seen < 168))
        self.assertEqual(runs[-1].active_model, ev.LOCAL)
        self.assertAlmostEqual(runs[-1].local_bytes, 20e6)
        start = runs[-1].hour_ms
        pred = ev.local_predictor(lambda x: 10e6, runs[-1].params, start)
        self.assertAlmostEqual(project(hist, start, 100e6, pred).reach_at_ms, start + 5 * HOUR_MS)


class EvaluateEndToEndTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hourly = synthetic_hourly(devices=6, hours=24 * 9)
        cls.hourly["device_id"] = cls.hourly["device_id"].map(lambda d: f"secret-{d}")
        known = {}
        for d, g in cls.hourly.dropna(subset=["mobile_bytes"]).groupby("device_id"):
            known[d] = dict(zip(g["hour_start_ms"].astype(int), g["mobile_bytes"].astype(float)))
        cls.report = ev.evaluate(cls.hourly, known, quotas_mb=(50, 300), fixed_params=PARAMS)
        cls.report["importance"] = ev.final_model_importance(cls.hourly, PARAMS)

    def test_every_method_is_scored_on_the_same_rows_and_cycles(self):
        r = self.report
        ns = {r["next_hour"][m]["n"] for m in ev.NEXT_HOUR_METHODS}
        self.assertEqual(ns, {r["rows"]})
        for q in ("50", "300", "all"):
            self.assertEqual(len({r["pet"][q][m]["n"] for m in ev.PET_METHODS}), 1)
        self.assertGreater(r["pet"]["50"]["global"]["n"], 0)
        c = r["pet_cycle_counts"]["50"]
        self.assertEqual(c["scored"] + c["crossing_unobserved"] + c["unknown_gap"] + c["never_crossed"], c["start_hours"])

    def test_outputs_are_written_without_device_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = ev.write_outputs(self.report, Path(tmp))
            for name in ("report.json", "report.md", "pet_mae_by_quota.png", "feature_importance.png"):
                self.assertIn(name, written)
                self.assertTrue((Path(tmp) / name).stat().st_size > 0)
            text = (Path(tmp) / "report.json").read_text(encoding="utf-8") + (Path(tmp) / "report.md").read_text(encoding="utf-8")
            self.assertNotIn("secret-", text)
            self.assertIn("D01", text)
            json.loads((Path(tmp) / "report.json").read_text(encoding="utf-8"))

    def test_importance_covers_every_feature(self):
        self.assertEqual(len(self.report["importance"]["gain_share"]), 12)


if __name__ == "__main__":
    unittest.main()
