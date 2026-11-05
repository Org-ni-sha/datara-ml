"""Tests for the phantom-spike repair, using values from the 2026-11-05 investigation."""

import unittest

import pandas as pd

from datara_ml.phantom_spikes import HOUR_MS, audit, clean_hourly_mobile, summary

MB = 1_000_000
H0 = 1_793_530_800_000  # an exact hour boundary (epoch ms)


def usage_row(device, offset_s, mb, net="LTE", status="OK", schema=3):
    return {
        "device_id": device,
        "timestamp": H0 + offset_s * 1000,
        "bytes_total": int(mb * MB),
        "network_type": net,
        "measurement_status": status,
        "schema_version": schema,
    }


def app_row(device, offset_s, mb, net="MOBILE", status="OK"):
    return {
        "device_id": device,
        "timestamp": H0 + offset_s * 1000,
        "bytes_total": int(mb * MB),
        "network_type": net,
        "measurement_status": status,
    }


def run(usage_rows, app_rows):
    usage = pd.DataFrame(usage_rows)
    apps = pd.DataFrame(app_rows) if app_rows else pd.DataFrame(
        columns=["device_id", "timestamp", "bytes_total", "network_type", "measurement_status"]
    )
    return clean_hourly_mobile(usage, apps).set_index(["device_id", "hour_start_ms"])


class PhantomSpikeTest(unittest.TestCase):

    def test_wifi_to_lte_spike_is_replaced_with_nsm(self):
        # f05ba8ad: 27,637 MB in 30 s vs NSM 77.7 MB for the hour
        h = run(
            [usage_row("a", 0, 0, net="WIFI"), usage_row("a", 30, 27637.147722),
             usage_row("a", 60, 2.0)],
            [app_row("a", 600, 77.710678)],
        ).loc[("a", H0)]
        self.assertEqual(h["action"], "replace_nsm")
        self.assertEqual(h["reconnect_spikes"], 1)
        self.assertAlmostEqual(h["mobile_bytes"] / MB, 77.710678, places=3)

    def test_none_to_lte_spike_is_replaced(self):
        # acdf17f5: 9,549 MB after "no connection" vs NSM 118 MB
        h = run(
            [usage_row("b", 0, 0, net="NONE"), usage_row("b", 30, 9549.916297)],
            [app_row("b", 600, 118.064512)],
        ).loc[("b", H0)]
        self.assertEqual(h["action"], "replace_nsm")

    def test_lte_to_2g_spike_is_replaced(self):
        # be83fd1d: 692 MB on 2G vs NSM 4.2 MB
        h = run(
            [usage_row("c", 0, 1, net="LTE"), usage_row("c", 30, 691.923682, net="2G")],
            [app_row("c", 600, 4.191466)],
        ).loc[("c", H0)]
        self.assertEqual(h["action"], "replace_nsm")

    def test_real_heavy_usage_below_nsm_is_kept(self):
        # ed4c65bc: 328.5 MB reading in an hour NSM measured at 487 MB
        h = run(
            [usage_row("d", 0, 0, net="WIFI"), usage_row("d", 30, 328.501809),
             usage_row("d", 60, 150.0)],
            [app_row("d", 600, 487.015932)],
        ).loc[("d", H0)]
        self.assertEqual(h["action"], "keep")
        self.assertAlmostEqual(h["mobile_bytes"] / MB, 478.501809, places=3)

    def test_borderline_close_to_nsm_is_kept(self):
        # 23d746a8: 55.1 MB vs NSM 51.1 MB, within 1.5x + 5 MB
        h = run(
            [usage_row("e", 0, 0, net="NONE"), usage_row("e", 30, 55.133257)],
            [app_row("e", 600, 51.110026)],
        ).loc[("e", H0)]
        self.assertEqual(h["action"], "keep")

    def test_spike_without_nsm_drops_the_hour(self):
        # 23d746a8: 520 MB with no NSM record for the hour
        h = run(
            [usage_row("f", 0, 0, net="WIFI"), usage_row("f", 30, 520.400712)],
            [],
        ).loc[("f", H0)]
        self.assertEqual(h["action"], "drop")
        self.assertTrue(pd.isna(h["mobile_bytes"]))

    def test_no_nsm_and_no_spike_is_kept(self):
        h = run([usage_row("g", 0, 3.0), usage_row("g", 30, 4.0)], []).loc[("g", H0)]
        self.assertEqual(h["action"], "keep")
        self.assertAlmostEqual(h["mobile_bytes"] / MB, 7.0)

    def test_wifi_only_nsm_hour_counts_as_zero_mobile(self):
        # NSM was read (a WIFI app row exists) and found 0 mobile bytes: the 40 MB is phantom.
        h = run(
            [usage_row("h", 0, 0, net="WIFI"), usage_row("h", 30, 40.0)],
            [app_row("h", 600, 30.0, net="WIFI")],
        ).loc[("h", H0)]
        self.assertEqual(h["action"], "replace_nsm")
        self.assertEqual(h["mobile_bytes"], 0)

    def test_small_spike_below_50mb_is_caught_by_hour_rule(self):
        # 45 MB phantom (not a "spike suspect") vs NSM 3 MB: hour rule still replaces it
        h = run(
            [usage_row("i", 0, 0, net="WIFI"), usage_row("i", 30, 45.0)],
            [app_row("i", 600, 3.0)],
        ).loc[("i", H0)]
        self.assertEqual(h["reconnect_spikes"], 0)
        self.assertEqual(h["action"], "replace_nsm")

    def test_two_spikes_in_one_hour_replace_once(self):
        # f05ba8ad: 1,112 and 1,118 MB in the same hour, NSM 77.4 MB
        h = run(
            [usage_row("j", 0, 0, net="NONE"), usage_row("j", 30, 1112.024488),
             usage_row("j", 2000, 0, net="NONE"), usage_row("j", 2030, 1118.374578)],
            [app_row("j", 600, 40.0), app_row("j", 2400, 37.441901)],
        ).loc[("j", H0)]
        self.assertEqual(h["reconnect_spikes"], 2)
        self.assertEqual(h["action"], "replace_nsm")
        self.assertAlmostEqual(h["mobile_bytes"] / MB, 77.441901, places=3)

    def test_schema_4_hours_are_never_changed(self):
        h = run(
            [usage_row("k", 0, 0, net="WIFI", schema=4), usage_row("k", 30, 900.0, schema=4)],
            [app_row("k", 600, 5.0)],
        ).loc[("k", H0)]
        self.assertEqual(h["action"], "keep")
        self.assertEqual(h["reconnect_spikes"], 0)

    def test_wifi_and_non_ok_rows_are_not_counted(self):
        h = run(
            [usage_row("l", 0, 500.0, net="WIFI"), usage_row("l", 30, 800.0, status="COUNTER_RESET"),
             usage_row("l", 60, 900.0, status="COUNTER_JUMP"), usage_row("l", 90, 2.0)],
            [app_row("l", 600, 2.0)],
        ).loc[("l", H0)]
        self.assertEqual(h["ts_mobile_bytes"], 2 * MB)
        self.assertEqual(h["action"], "keep")

    def test_hours_and_devices_are_separate(self):
        h = run(
            [usage_row("m", 0, 0, net="WIFI"), usage_row("m", 30, 1000.0),
             usage_row("m", 3600 + 30, 5.0), usage_row("n", 30, 6.0)],
            [app_row("m", 600, 10.0), app_row("m", 3600 + 600, 5.0), app_row("n", 600, 6.0)],
        )
        self.assertEqual(h.loc[("m", H0), "action"], "replace_nsm")
        self.assertEqual(h.loc[("m", H0 + HOUR_MS), "action"], "keep")
        self.assertEqual(h.loc[("n", H0), "action"], "keep")

    def test_audit_and_summary_list_only_changed_hours(self):
        usage = pd.DataFrame([usage_row("o", 0, 0, net="WIFI"), usage_row("o", 30, 2000.0),
                              usage_row("p", 30, 1.0)])
        apps = pd.DataFrame([app_row("o", 600, 20.0), app_row("p", 600, 1.0)])
        hourly = clean_hourly_mobile(usage, apps)
        a = audit(hourly)
        self.assertEqual(list(a["device_id"]), ["o"])
        self.assertAlmostEqual(a.iloc[0]["mobile_mb"], 20.0)
        self.assertIn("hours replaced with NSM:     1", summary(hourly))
        self.assertIn("MB removed:                  1,980.0", summary(hourly))


if __name__ == "__main__":
    unittest.main()
