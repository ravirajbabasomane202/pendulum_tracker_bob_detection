import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import JOBS, app
import dashboard as dash


def synthetic_dataset(n=1800, fs=30.0, noise=0.5, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = 140 * np.exp(-0.004 * t) * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0, noise, n)
    y = 55 * np.exp(-0.004 * t) * np.sin(2 * np.pi * 0.5 * t + 0.7) + rng.normal(0, noise, n)
    return t, x, y


def make_job(jid="dash-test", n=1800, fs=30.0):
    t, x, y = synthetic_dataset(n, fs)
    raw = np.column_stack((x + 320, -y + 240))
    JOBS[jid] = dict(state="done", name="synthetic.mp4", x=x, y=y, unit="px", i0=0, i1=n, fps=fs,
                     raw_coordinates=raw, stabilized_coordinates=raw + 3, zoom_coordinates=raw + 3,
                     params={"g": 9.81, "length": None}, summary={"tracking_coverage": 0.99})
    return jid


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jid = make_job()
        cls.c = app.test_client()

    def get(self, path, **q):
        return self.c.get(path, query_string=q)

    def test_pages_and_datasets(self):
        self.assertEqual(self.get("/dashboard").status_code, 200)
        self.assertEqual(self.get(f"/dashboard/{self.jid}").status_code, 200)
        self.assertEqual(self.get("/static/plotly.min.js").status_code, 200)
        ids = [d["id"] for d in self.get("/api/dashboard/datasets").get_json()["datasets"]]
        self.assertIn(self.jid, ids)
        meta = self.get(f"/api/dashboard/{self.jid}/meta").get_json()
        self.assertEqual({s["key"] for s in meta["series"]}, {"analysis", "raw", "stab", "zoom"})
        self.assertEqual(self.get("/api/dashboard/nope/meta").status_code, 404)

    def test_every_filter_and_outlier_method(self):
        for f in dash.FILTER_NAMES:
            for o in dash.OUTLIER_METHODS:
                r = self.get(f"/api/dashboard/{self.jid}/analysis", filter=f, outlier=o, window=15, cutoff=1.5,
                             cutoff_high=3.0)
                self.assertEqual(r.status_code, 200, (f, o, r.data[:200]))
                j = r.get_json()
                self.assertEqual(len(j["view"]["x"]), len(j["view"]["t"]))

    def test_physics_is_recovered(self):
        j = self.get(f"/api/dashboard/{self.jid}/analysis", filter="sg").get_json()
        P, A = j["periods"], j["amp"]
        self.assertAlmostEqual(P["fx_fft"], 0.8, delta=0.02)
        self.assertAlmostEqual(P["fy_fft"], 0.5, delta=0.02)
        self.assertAlmostEqual(P["fx_peak"], 0.8, delta=0.05)
        self.assertAlmostEqual(A["gamma"], 0.004, delta=0.002)
        self.assertAlmostEqual(A["Ax"], 140, delta=10)
        self.assertGreater(j["amp"]["Q"], 10)
        for row in j["damping"]["rows"]:
            self.assertIsNotNone(row["gamma"], row)

    def test_filters_remove_the_noise_but_keep_the_signal(self):
        # synthetic noise sigma = 0.5 px: the residual (filtered - original) must be about that size
        none = self.get(f"/api/dashboard/{self.jid}/analysis", filter="none").get_json()
        self.assertEqual(none["resid"]["x"]["std"], 0)
        for f in ("sg", "gauss", "ma", "median", "lowess"):
            j = self.get(f"/api/dashboard/{self.jid}/analysis", filter=f, window=11).get_json()
            # smoothing removes noise but also shaves a little off the 140 px swing (window-dependent)
            self.assertTrue(0.3 < j["resid"]["x"]["std"] < 15, (f, j["resid"]["x"]["std"]))
            self.assertGreater(j["amp"]["Ax"], 118, f)
        sg = self.get(f"/api/dashboard/{self.jid}/analysis", filter="sg", window=11).get_json()
        self.assertTrue(0.3 < sg["resid"]["x"]["std"] < 0.9, sg["resid"]["x"]["std"])   # SG follows the signal

    def test_outlier_removal_replaces_spikes(self):
        t, x, y = synthetic_dataset()
        x2 = x.copy()
        x2[[100, 500, 900]] += 400
        out, count = dash.remove_outliers(x2, "hampel", 3.0, [], "x")
        self.assertGreaterEqual(count, 3)
        self.assertLess(np.abs(out[[100, 500, 900]] - x[[100, 500, 900]]).max(), 8)
        self.assertLess(count, 40, "a clean signal must not be flagged wholesale")
        for m in ("zscore", "iqr"):
            w = []
            o, c = dash.remove_outliers(x2, m, 3.0, w, "x")
            self.assertGreaterEqual(c, 3, (m, w))
            self.assertLess(np.abs(o[[100, 500, 900]] - x[[100, 500, 900]]).max(), 8, m)
        _, c = dash.remove_outliers(x, "hampel", 3.0, [], "x")
        self.assertLess(c, 0.05 * len(x), "clean data must rarely be flagged")

    def test_butterworth_cutoff_warning(self):
        j = self.get(f"/api/dashboard/{self.jid}/analysis", filter="butter_lp", cutoff=40).get_json()
        self.assertTrue(any("Nyquist" in w for w in j["warnings"]))

    def test_json_has_no_nan_and_exports_work(self):
        r = self.get(f"/api/dashboard/{self.jid}/analysis", filter="sg")
        json.loads(r.data.decode(), parse_constant=lambda c: self.fail("NaN in JSON"))
        csv = self.get(f"/api/dashboard/{self.jid}/filtered.csv", filter="ma")
        self.assertEqual(csv.status_code, 200)
        self.assertGreater(len(csv.data.decode().splitlines()), 1000)
        z = self.get(f"/api/dashboard/{self.jid}/images.zip", filter="sg")
        self.assertEqual(z.status_code, 200)
        names = zipfile.ZipFile(io.BytesIO(z.data)).namelist()
        self.assertEqual(len(names), 13)
        ppt = self.get(f"/api/dashboard/{self.jid}/report.pptx", filter="sg")
        self.assertEqual(ppt.status_code, 200)
        from pptx import Presentation
        self.assertGreaterEqual(len(Presentation(io.BytesIO(ppt.data)).slides), 14)

    def test_csv_and_json_upload_roundtrip(self):
        csv_text = self.get_export_csv()
        r = self.c.post("/api/dashboard/upload", data={"file": (io.BytesIO(csv_text.encode()), "trajectory_full.csv")})
        self.assertEqual(r.status_code, 200, r.data)
        did = r.get_json()["id"]
        j = self.get(f"/api/dashboard/{did}/analysis").get_json()
        self.assertAlmostEqual(j["periods"]["fx_fft"], 0.8, delta=0.03)
        bad = self.c.post("/api/dashboard/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")})
        self.assertEqual(bad.status_code, 400)
        payload = {"unit": "px", "params": {}, "summary": {}, "time_series": {
            "t": list(np.arange(300) / 30.0), "x": list(np.sin(np.arange(300) / 5.0) * 50),
            "y": list(np.cos(np.arange(300) / 7.0) * 20)}}
        r = self.c.post("/api/dashboard/upload", data={"file": (io.BytesIO(json.dumps(payload).encode()), "e.json")})
        self.assertEqual(r.status_code, 200, r.data)

    def test_short_recording_does_not_crash(self):
        make_job("short", n=90)
        r = self.get("/api/dashboard/short/analysis", filter="sg")
        self.assertEqual(r.status_code, 200, r.data[:300])
        self.assertEqual(self.get("/api/dashboard/short/report.pptx").status_code, 200)

    def get_export_csv(self):
        # build a CSV in the same format as the main app's /csv route
        t, x, y = synthetic_dataset()
        lines = ["# Object Trajectory Analysis — full export", "# param_unit,px", "# param_g,9.81",
                 "# summary_tracking_coverage,1.0", "",
                 "time_s,x_px,y_px,raw_x_px,raw_y_px,stabilized_x_px,stabilized_y_px,zoom_normalized_x_px,"
                 "zoom_normalized_y_px,z_est,r,phi_deg,A_envelope_px"]
        for a, b, c in zip(t, x, y):
            lines.append(f"{a:.4f},{b:.4f},{c:.4f},{b + 320:.3f},{240 - c:.3f},{b + 320:.3f},{240 - c:.3f},"
                         f"{b + 320:.3f},{240 - c:.3f},0,0,0,")
        lines += ["", "# sliding window (θ, periods)", "t_s,Tx_s"]
        return "\n".join(lines)


if __name__ == "__main__":
    unittest.main()
