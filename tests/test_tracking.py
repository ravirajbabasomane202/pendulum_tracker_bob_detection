import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import JOBS, app, clean, run_job, track


class BluePendulumTrackerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="pendulum-tracking-test-"))
        cls.video_path = cls.temp_dir / "moving-blue-bob.avi"
        cls.width, cls.height, cls.frame_count, cls.fps = 360, 600, 220, 30
        cls.expected_raw = np.zeros((cls.frame_count, 2), dtype=np.float64)
        cls.expected_world = np.zeros_like(cls.expected_raw)
        cls.radii = np.linspace(12, 34, cls.frame_count)

        rng = np.random.default_rng(4)
        background = np.full((cls.height, cls.width, 3), 225, np.uint8)
        for y in range(10, cls.height, 32):
            cv2.line(background, (0, y), (cls.width - 1, y), (194, 202, 206), 1)
        for x in range(10, cls.width, 34):
            cv2.line(background, (x, 0), (x, cls.height - 1), (202, 207, 208), 1)
        for _ in range(180):
            x, y = int(rng.integers(8, cls.width - 8)), int(rng.integers(8, cls.height - 8))
            color = int(rng.integers(60, 185))
            cv2.circle(background, (x, y), int(rng.integers(2, 5)), (color, color, color), -1)
        for y in (180, 430):
            cv2.rectangle(background, (168, y), (182, y + 34), (42, 67, 117), -1)

        blue = tuple(int(v) for v in cv2.cvtColor(
            np.uint8([[[112, 225, 225]]]), cv2.COLOR_HSV2BGR
        )[0, 0])
        writer = cv2.VideoWriter(
            str(cls.video_path), cv2.VideoWriter_fourcc(*"MJPG"),
            cls.fps, (cls.width, cls.height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the synthetic test video.")

        for i in range(cls.frame_count):
            phase = 2 * math.pi * i / 90
            world_x = 180 + 75 * math.sin(phase)
            world_y = 310 + 45 * math.cos(phase * 0.93)
            cam_x = 35 * math.sin(2 * math.pi * i / 210)
            cam_y = 20 * math.cos(2 * math.pi * i / 170)
            affine = np.float32([[1, 0, cam_x], [0, 1, cam_y]])
            frame = cv2.warpAffine(
                background, affine, (cls.width, cls.height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(215, 215, 215),
            )
            x, y = int(round(world_x + cam_x)), int(round(world_y + cam_y))
            radius = int(round(cls.radii[i]))
            cv2.circle(frame, (x, y), radius, blue, -1, cv2.LINE_AA)
            cv2.circle(frame, (x - radius // 3, y - radius // 3),
                       max(2, radius // 5), (245, 248, 250), -1, cv2.LINE_AA)
            # A rust-red leg crosses the blue ball for several frames.
            if 43 <= i <= 49:
                leg_x = int(round(180 + cam_x))
                cv2.rectangle(frame, (leg_x - 4, y - radius), (leg_x + 4, y + radius),
                              (38, 62, 133), -1)
            writer.write(frame)
            cls.expected_raw[i] = (x, y)
            # Stabilized coordinates are expressed in the first frame's image basis.
            cls.expected_world[i] = (world_x, world_y + 20)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_color_tracking_camera_motion_occlusion_and_size_range(self):
        raw, stabilized, zoom_normalized, sizes, fps, width, height, info = track(
            str(self.video_path), stabilize=True
        )
        self.assertEqual((width, height, round(fps)), (self.width, self.height, self.fps))
        valid = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1])
        error = np.linalg.norm(raw[valid] - self.expected_raw[valid], axis=1)
        self.assertGreaterEqual(float(np.mean(error <= 10)), 0.95)
        self.assertGreater(float(np.std(raw[valid, 0])), 40.0, "the path must follow the swing, not a fixed chair")
        stable_error = np.linalg.norm(stabilized[valid] - self.expected_world[valid], axis=1)
        self.assertLess(float(np.median(stable_error)), 8.0, "camera motion should be removed")
        self.assertTrue(np.isfinite(zoom_normalized[valid]).all())
        self.assertEqual(info["calibration_source"], "auto")
        self.assertLessEqual(info["hsv_limits"]["h_max"] - info["hsv_limits"]["h_min"], 25)
        self.assertTrue(np.isfinite(sizes[10]))
        self.assertTrue(np.isfinite(sizes[-10]))
        self.assertLess(float(sizes[10]), 45)
        self.assertGreater(float(sizes[-10]), 50)

    def test_click_colour_override(self):
        click_x, click_y = self.expected_raw[0]
        raw, _, _, _, _, _, _, info = track(
            str(self.video_path), ball_click=(click_x, click_y), stabilize=True
        )
        self.assertEqual(info["calibration_source"], "click")
        self.assertGreater(np.isfinite(raw[:, 0]).sum(), 180)

    def test_clean_interpolates_through_fifteen_missing_frames(self):
        count = 180
        frames = np.arange(count)
        pos = np.column_stack((50 + 20 * np.sin(frames / 8), 100 + 12 * np.cos(frames / 9))).astype(float)
        size = np.full(count, 24.0)
        pos[70:85] = np.nan
        size[70:85] = np.nan
        start, end = clean(pos, size, 30)
        self.assertEqual((start, end), (0, count))
        self.assertTrue(np.isfinite(pos[70:85]).all())
        self.assertTrue(np.isfinite(size[70:85]).all())

    def test_existing_analysis_and_export_routes_keep_legacy_coordinates(self):
        job_id = "synthetic-export-check"
        JOBS[job_id] = {"state": "running", "pct": 0, "stage": "test"}
        run_job(job_id, str(self.video_path), {
            "h_min": "100", "h_max": "130", "s_min": "120", "v_min": "50",
            "camera_stabilize": "1", "zoom_normalize": "1", "prefer": "auto",
        })
        job = JOBS[job_id]
        self.assertEqual(job["state"], "done", job.get("error"))
        result = job["result"]
        self.assertEqual(result["px"], result["raw_x"], "video overlay coordinates must remain raw frame pixels")
        self.assertTrue(result["summary"]["shape"])

        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        csv_response = client.get(f"/csv/{job_id}")
        json_response = client.get(f"/json/{job_id}")
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(json_response.status_code, 200)
        csv_rows = csv_response.data.decode("utf-8").splitlines()
        header = next(row.split(",") for row in csv_rows if row.startswith("time_s,"))
        self.assertEqual(header[:3], ["time_s", "x_px", "y_px"])
        self.assertIn("raw_x_px", header)
        self.assertIn("stabilized_x_px", header)
        exported = json_response.get_json()["time_series"]
        for key in ("x", "y", "raw_x", "stabilized_x", "zoom_normalized_x"):
            self.assertIn(key, exported)
        debug_response = client.get(f"/debug/{job_id}")
        samples_response = client.get(f"/debug-samples/{job_id}")
        self.assertEqual(debug_response.status_code, 200)
        self.assertEqual(samples_response.status_code, 200)
        debug_response.close()
        samples_response.close()


if __name__ == "__main__":
    unittest.main()