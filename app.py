"""Pendulum Tracker - Flask backend.
Upload a video -> OpenCV tracks the bob -> x(t), y(t), Tx/Ty/fx/fy, delta-f, theta(t)
are computed with numpy/scipy -> the browser replays them in sync with the video.
Run:  python app.py   then open http://127.0.0.1:5000
"""
import csv
import io
import os
import threading
import uuid

import cv2
import numpy as np
from flask import Flask, abort, Response, jsonify, render_template, request, send_from_directory
from analysis import (amplitude, amplitude_decay, classify_shape, envelope_and_damping, fft_peak,  # noqa: F401
                      L, _fill_nan, nanmean2, peak_period, pca, phase_diff_deg, safe, sliding, zc_period)
from dashboard import dashboard_bp

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "uploads")
os.makedirs(UPLOADS, exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
JOBS = {}
app.config["JOBS"] = JOBS
app.register_blueprint(dashboard_bp)


# ---------------------------------------------------------------- tracking
DEFAULT_HSV = {"h_min": 100, "h_max": 130, "s_min": 120, "v_min": 50}
MAX_INTERPOLATED_MISSING_FRAMES = 15
MAX_ANALYSIS_GAP_SECONDS = 4.0   # gaps up to this long are interpolated for the analysis / overlay


def _color_masks(hsv, limits, detection_scale=1.0):
    """Build a colour mask for the learned bob HSV range and fill enclosed holes."""
    h_min, h_max = int(limits["h_min"]), int(limits["h_max"])
    s_min, v_min = int(limits["s_min"]), int(limits["v_min"])
    if limits.get("hue_wrap") or h_min > h_max:
        # Red-style wrap: two ranges
        mask1 = cv2.inRange(hsv, np.array([0, s_min, v_min], np.uint8),
                            np.array([h_max, 255, 255], np.uint8))
        mask2 = cv2.inRange(hsv, np.array([h_min, s_min, v_min], np.uint8),
                            np.array([179, 255, 255], np.uint8))
        color_mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array([h_min, s_min, v_min], np.uint8)
        upper = np.array([h_max, 255, 255], np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)
    opened = cv2.morphologyEx(
        color_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    close_size = max(5, int(round(9 * detection_scale)))
    if close_size % 2 == 0:
        close_size += 1
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    filled = np.zeros_like(closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return color_mask, filled


def _blob_candidates(hsv, limits, detection_scale=1.0, relaxed=False, shape_relax=1.0):
    """Find connected components in the learned bob colour range and score by shape/colour.

    `shape_relax` > 1 lowers circularity/solidity thresholds so non-round bobs
    (cubes, irregular shapes) can still be accepted after user confirmation.
    """
    raw_mask, filled_mask = _color_masks(hsv, limits, detection_scale)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(filled_mask, 8)
    accepted, rejected = [], []
    height, width = filled_mask.shape
    hue_center = (limits["h_min"] + limits["h_max"]) / 2.0
    hue_half_range = max(1.0, (limits["h_max"] - limits["h_min"]) / 2.0)

    for label in range(1, count):
        x, y, w, h, pixel_area = [int(v) for v in stats[label]]
        if pixel_area <= 0:
            continue
        box = (x, y, w, h)
        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        circularity = float(np.clip(4.0 * np.pi * pixel_area / (perimeter * perimeter + 1e-9), 0, 1))
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = float(np.clip(pixel_area / (hull_area + 1e-9), 0, 1))
        aspect = max(w, h) / max(1, min(w, h))
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height

        color_crop = raw_mask[y:y + h, x:x + w] > 0
        hsv_crop = hsv[y:y + h, x:x + w]
        samples = hsv_crop[color_crop]
        if not len(samples):
            samples = hsv_crop[component > 0]
        median_hsv = np.median(samples, axis=0) if len(samples) else np.array([hue_center, 0, 0])
        h_med, s_med, v_med = [float(v) for v in median_hsv]
        hue_delta = abs(h_med - hue_center)
        hue_delta = min(hue_delta, 180.0 - hue_delta)
        hue_score = float(np.clip(1.0 - hue_delta / max(12.0, hue_half_range * 1.5), 0, 1))
        sat_score = float(np.clip((s_med - limits["s_min"]) / max(1.0, 255 - limits["s_min"]), 0, 1))
        val_score = float(np.clip((v_med - limits["v_min"]) / max(1.0, 255 - limits["v_min"]), 0, 1))
        color_score = 0.62 * hue_score + 0.25 * sat_score + 0.13 * val_score
        why = None
        sr = max(float(shape_relax), 1.0)
        if relaxed:   # blurred (focus shift) / zoomed / partly hidden ball
            min_circularity = (0.25 if touches_border else 0.32) / sr
            min_solidity = (0.35 if touches_border else 0.42) / sr
            max_aspect = (3.0 if touches_border else 2.6) * min(sr, 2.0)
        else:
            min_circularity = (0.48 if touches_border else 0.70) / sr
            min_solidity = (0.60 if touches_border else 0.70) / sr
            max_aspect = (1.9 if touches_border else 1.6) * min(sr, 2.0)
        min_circularity = max(0.08, min_circularity)
        min_solidity = max(0.15, min_solidity)
        if aspect > max_aspect:
            why = f"aspect ratio {aspect:.2f} > {max_aspect:.1f}"
        elif circularity < min_circularity:
            why = f"circularity {circularity:.2f} < {min_circularity:.2f}"
        elif solidity < min_solidity:
            why = f"solidity {solidity:.2f} < {min_solidity:.2f}"

        candidate = dict(
            x=float(centers[label][0]), y=float(centers[label][1]),
            area=float(pixel_area), radius=float(np.sqrt(pixel_area / np.pi)),
            circularity=circularity, solidity=solidity, aspect=float(aspect),
            color_score=color_score, hsv=(h_med, s_med, v_med),
            border=bool(touches_border), box=box,
        )
        if why:
            rejected.append((box, why, candidate))
        else:
            if touches_border:
                candidate["border_penalty"] = 0.35
            else:
                candidate["border_penalty"] = 0.0
            accepted.append(candidate)
    return accepted, rejected, raw_mask


def _adaptive_hsv(sample, defaults, wide=False):
    """Narrow hue around a sampled bob colour and adapt S/V to that sample.

    Handles red hue wrap-around (OpenCV hue 0..179). When `wide` is True the
    hue window is expanded for more forgiving tracking of non-uniform objects.
    """
    h, s, v = [float(q) for q in sample]
    half = 18 if wide else 12
    h_min = int(round(h - half))
    h_max = int(round(h + half))
    # Normal non-wrapping case
    if h_min >= 0 and h_max <= 179:
        return {
            "h_min": h_min,
            "h_max": h_max,
            "s_min": max(25, min(int(defaults.get("s_min", 80)), int(round(s * 0.55)))),
            "v_min": max(15, min(int(defaults.get("v_min", 40)), int(round(v * 0.40)))),
        }
    # Wrap-around (typically red): store a flag range that _color_masks understands
    # by using the convention h_min > h_max means "wrap".
    return {
        "h_min": (h_min + 180) % 180,
        "h_max": h_max % 180,
        "s_min": max(25, min(int(defaults.get("s_min", 80)), int(round(s * 0.55)))),
        "v_min": max(15, min(int(defaults.get("v_min", 40)), int(round(v * 0.40)))),
        "hue_wrap": True,
    }


def _click_color(cap, click, frame_width, frame_height, detection_scale, defaults):
    """Sample a small, saturated colour neighborhood around a clicked first-frame point.

    Works for any bob colour (not limited to blue). Prefers pixels with similar hue
    and reasonable saturation/value near the click.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        raise ValueError("Could not read the first frame for click calibration.")
    if detection_scale < 1:
        frame = cv2.resize(frame, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x = int(round(float(click[0]) * detection_scale))
    y = int(round(float(click[1]) * detection_scale))
    if not (0 <= x < hsv.shape[1] and 0 <= y < hsv.shape[0]):
        raise ValueError("The clicked calibration point is outside the video frame.")
    radius = max(4, int(round(12 * detection_scale)))
    roi = hsv[max(0, y - radius):min(hsv.shape[0], y + radius + 1),
              max(0, x - radius):min(hsv.shape[1], x + radius + 1)]
    sat = roi[:, :, 1].astype(np.int16)
    val = roi[:, :, 2].astype(np.int16)
    hue = roi[:, :, 0].astype(np.int16)
    center_hue = int(hsv[y, x, 0])
    delta = np.abs(hue - center_hue)
    delta = np.minimum(delta, 180 - delta)
    # Prefer saturated, similar-hue pixels; fall back to any nearby non-dark pixels
    use = (sat >= 40) & (val >= 30) & (delta <= 20)
    if int(use.sum()) < 5:
        use = (sat >= 25) & (val >= 25) & (delta <= 30)
    if int(use.sum()) < 3:
        use = (val >= 20) & (delta <= 40)
    if not np.any(use):
        # Last resort: just the centre pixel neighbourhood
        use = np.ones(roi.shape[:2], dtype=bool)
    return np.median(roi[use], axis=0)


def _sample_bbox_color(cap, bbox, frame_width, frame_height, detection_scale):
    """Sample median HSV from a user- or auto-selected bounding box on the first frame."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    if not ok:
        raise ValueError("Could not read the first frame for bob selection.")
    if detection_scale < 1:
        frame = cv2.resize(frame, None, fx=detection_scale, fy=detection_scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x, y, w, h = [int(round(v * detection_scale)) for v in bbox]
    x = max(0, min(x, hsv.shape[1] - 1))
    y = max(0, min(y, hsv.shape[0] - 1))
    w = max(2, min(w, hsv.shape[1] - x))
    h = max(2, min(h, hsv.shape[0] - y))
    roi = hsv[y:y + h, x:x + w]
    # Prefer more saturated pixels inside the box (bob body over highlight/background)
    sat = roi[:, :, 1]
    val = roi[:, :, 2]
    mask = (sat >= 30) & (val >= 25)
    if int(mask.sum()) < max(8, (w * h) // 20):
        mask = val >= 20
    if not np.any(mask):
        return np.median(roi.reshape(-1, 3), axis=0)
    return np.median(roi[mask], axis=0)


def _background_feature_mask(hsv, bob_mask):
    """Exclude the bob and common warm/red distractors from camera feature tracking."""
    warm = cv2.inRange(hsv, np.array([0, 70, 35], np.uint8), np.array([38, 255, 255], np.uint8))
    red = cv2.inRange(hsv, np.array([168, 70, 35], np.uint8), np.array([179, 255, 255], np.uint8))
    moving_objects = cv2.bitwise_or(bob_mask, cv2.bitwise_or(warm, red))
    moving_objects = cv2.dilate(
        moving_objects, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    )
    return cv2.bitwise_not(moving_objects)


def _estimate_camera_affine(prev_gray, gray, prev_feature_mask, feature_mask):
    """Estimate previous-to-current camera motion from background corners only."""
    points0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=350, qualityLevel=0.01, minDistance=7,
        blockSize=7, mask=prev_feature_mask,
    )
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if points0 is None or len(points0) < 8:
        return identity, False
    points1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, points0, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    if points1 is None or status is None:
        return identity, False
    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    good = status.reshape(-1) == 1
    p0, p1 = p0[good], p1[good]
    if len(p1):
        ix = np.clip(np.rint(p1[:, 0]).astype(int), 0, gray.shape[1] - 1)
        iy = np.clip(np.rint(p1[:, 1]).astype(int), 0, gray.shape[0] - 1)
        keep = feature_mask[iy, ix] > 0
        p0, p1 = p0[keep], p1[keep]
    if len(p1) < 8:
        return identity, False
    affine, inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=150, confidence=0.99, refineIters=8,
    )
    if affine is None or inliers is None or int(inliers.sum()) < 8:
        return identity, False
    scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
    translation = float(np.hypot(affine[0, 2], affine[1, 2]))
    if (not np.isfinite(scale) or scale < 0.82 or scale > 1.22
            or translation > 0.20 * max(gray.shape)):
        return identity, False
    return affine.astype(np.float64), True


def _transform_point(matrix, point):
    p = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
    return (matrix @ p)[:2]


def _choose_candidate(candidates, previous, velocity, previous_radius, gap, shape, prefer,
                      world_to_frame=None, camera_scale=1.0, relax=1.0, anchor_world=None):
    """Score blue, shape-valid candidates against the prediction.

    `previous`/`velocity`/`previous_radius` live in stabilized WORLD coordinates while the
    candidates live in the CURRENT FRAME.  The prediction is therefore mapped into the current
    frame with `world_to_frame` (this is what keeps tracking alive during zoom / camera moves).
    """
    height, width = shape
    positional_weights = {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}
    best, best_score, best_distance = None, -np.inf, None
    cs = max(float(camera_scale), 1e-9)
    prediction = None
    if previous is not None:
        pw = previous + velocity * gap
        prediction = _transform_point(world_to_frame, pw) if world_to_frame is not None else pw
    anchor = None
    if previous is None and anchor_world is not None:
        anchor = _transform_point(world_to_frame, anchor_world) if world_to_frame is not None else anchor_world
    tracking_ok = gap <= MAX_INTERPOLATED_MISSING_FRAMES + 1
    for candidate in candidates:
        size_score = 0.5
        distance_score = 0.5
        distance = None
        if previous_radius and previous is not None and tracking_ok:
            world_radius = candidate["radius"] / cs
            change = abs(world_radius / previous_radius - 1.0)
            limit_change = min(0.35 * gap, 2.0) * relax
            if change > limit_change:
                candidate["tracking_reject"] = "radius change too large"
                continue
            size_score = float(np.clip(1.0 - change / max(limit_change, 1e-6), 0, 1))
        if prediction is not None and tracking_ok:
            distance = float(np.hypot(candidate["x"] - prediction[0], candidate["y"] - prediction[1]))
            pr = (previous_radius or 8.0) * cs
            speed = float(np.hypot(*velocity)) * cs
            motion_limit = relax * max(26.0, pr * 4.0 * np.sqrt(gap), speed * gap + pr * 1.5)
            if distance > motion_limit:
                candidate["tracking_reject"] = "outside predicted motion window"
                continue
            distance_score = float(np.clip(1.0 - distance / motion_limit, 0, 1))
        elif anchor is not None:      # lost for a long time: prefer blobs near where it vanished
            distance = float(np.hypot(candidate["x"] - anchor[0], candidate["y"] - anchor[1]))
            distance_score = float(np.clip(1.0 - distance / (0.8 * max(height, width)), 0, 1))
        round_score = float(np.clip(0.5 * candidate["circularity"] + 0.5 * candidate["solidity"], 0, 1))
        score = (0.48 * candidate["color_score"] + 0.20 * round_score
                 + 0.14 * size_score + 0.18 * distance_score - candidate["border_penalty"])
        tie = positional_weights.get(prefer)
        if tie:
            score += 0.015 * (tie[0] * candidate["x"] / max(1, width)
                              + tie[1] * candidate["y"] / max(1, height))
        candidate["score"] = float(score)
        candidate["prediction_distance"] = distance
        if score > best_score:
            best, best_score, best_distance = candidate, score, distance
    return best, best_distance


def _draw_path(image, positions, detection_scale=1.0, offset=(0, 0), point_limit=None):
    points = positions
    if point_limit is not None:
        points = positions[max(0, point_limit[0]):min(len(positions), point_limit[1])]
    scale = detection_scale
    paths = []
    active = []
    for point in points:
        if np.isfinite(point).all():
            active.append((int(round(point[0] * scale + offset[0])),
                           int(round(point[1] * scale + offset[1]))))
        elif len(active) > 1:
            paths.append(active)
            active = []
        else:
            active = []
    if len(active) > 1:
        paths.append(active)
    for path_points in paths:
        cv2.polylines(image, [np.asarray(path_points, np.int32).reshape(-1, 1, 2)],
                      False, (255, 190, 25), 2, cv2.LINE_AA)


def _annotate_debug_frame(image, candidates, rejected, chosen, raw_position, detection_scale):
    scale = 1.0 / max(detection_scale, 1e-9)
    font = 0.45 if image.shape[1] < 1000 else 0.65
    for box, reason, _candidate in rejected:
        x, y, w, h = [int(round(v * scale)) for v in box]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, "blue rejected: " + reason[:54], (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 80, 255), 1, cv2.LINE_AA)
    for candidate in candidates:
        x, y, w, h = [int(round(v * scale)) for v in candidate["box"]]
        is_chosen = candidate is chosen
        color = (0, 220, 30) if is_chosen else (255, 0, 0)
        reason = "BOB" if is_chosen else candidate.get("tracking_reject", "not selected")
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3 if is_chosen else 2)
        cv2.putText(image, reason[:54], (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, font, color if is_chosen else (0, 80, 255), 1, cv2.LINE_AA)
    if raw_position is not None and np.isfinite(raw_position).all():
        center = tuple(int(round(v)) for v in raw_position)
        cv2.circle(image, center, max(8, int(12 * font)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(image, center, (0, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)


def _write_debug_samples(path, samples, raw_pos, detection_scale):
    if not samples:
        return
    tile_w, tile_h, columns = 360, 320, 3
    rows = int(np.ceil(len(samples) / columns))
    sheet = np.zeros((rows * tile_h, columns * tile_w, 3), np.uint8)
    for slot, sample in enumerate(samples):
        image = sample["image"].copy()
        h, w = image.shape[:2]
        fit = min((tile_w - 12) / max(1, w), (tile_h - 38) / max(1, h))
        resized = cv2.resize(image, (max(1, int(w * fit)), max(1, int(h * fit))),
                             interpolation=cv2.INTER_AREA if fit < 1 else cv2.INTER_LINEAR)
        dx = (tile_w - resized.shape[1]) // 2
        dy = 28 + (tile_h - 28 - resized.shape[0]) // 2
        local_scale = fit
        chosen = sample["chosen"]
        for box, reason, _candidate in sample["rejected"]:
            x, y, bw, bh = [int(round(v * local_scale)) for v in box]
            cv2.rectangle(resized, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
            cv2.putText(resized, reason[:28], (x, max(12, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 70, 255), 1, cv2.LINE_AA)
        for candidate in sample["candidates"]:
            x, y, bw, bh = [int(round(v * local_scale)) for v in candidate["box"]]
            mine = candidate is chosen
            color = (0, 225, 25) if mine else (255, 0, 0)
            reason = "BOB" if mine else candidate.get("tracking_reject", "other blue blob")
            cv2.rectangle(resized, (x, y), (x + bw, y + bh), color, 2 if mine else 1)
            cv2.putText(resized, reason[:28], (x, max(12, y - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, color if mine else (0, 70, 255), 1, cv2.LINE_AA)
        frame_idx = sample["index"]
        first, last = max(0, frame_idx - 18), min(len(raw_pos), frame_idx + 19)
        _draw_path(resized, raw_pos, detection_scale * fit, offset=(0, 0), point_limit=(first, last))
        tile_x, tile_y = (slot % columns) * tile_w, (slot // columns) * tile_h
        sheet[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w] = (24, 31, 38)
        sheet[tile_y + dy:tile_y + dy + resized.shape[0],
              tile_x + dx:tile_x + dx + resized.shape[1]] = resized
        cv2.putText(sheet, f"Frame {frame_idx + 1}", (tile_x + 8, tile_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 240, 242), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def detect_bob_candidates(path, max_candidates=5):
    """Detect likely bob objects in the first frames of a video.

    Strategy (in order of preference):
    1. Motion blobs over the first ~1.5 s (pendulum bob usually moves).
    2. Saturated-colour circular blobs (legacy blue / coloured ball).
    3. Largest mid-frame non-border blobs as a last resort.

    Returns dict with first-frame JPEG (base64), list of candidate bboxes in
    original-frame coordinates, and the index of the best guess.
    """
    import base64
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("The uploaded video has no readable frames.")

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read the first frame.")
    first_bgr = first.copy()

    # --- motion accumulation over early frames ---
    n_motion = min(frame_count, max(8, int(round(fps * 1.5))))
    step = max(1, n_motion // 12)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    motion_acc = np.zeros((H, W), np.float32)
    for i in range(1, n_motion):
        ok, fr = cap.read()
        if not ok:
            break
        if i % step:
            continue
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, th = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        motion_acc += th.astype(np.float32)
        prev_gray = gray
    cap.release()

    candidates = []  # list of dicts: x,y,w,h,score,source,circularity,...

    # Motion-based candidates
    if motion_acc.max() > 0:
        motion_norm = (motion_acc / motion_acc.max() * 255).astype(np.uint8)
        _, mbin = cv2.threshold(motion_norm, 40, 255, cv2.THRESH_BINARY)
        mbin = cv2.morphologyEx(mbin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        cnts, _ = cv2.findContours(mbin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 80 or area > 0.25 * W * H:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 6 or h < 6:
                continue
            peri = cv2.arcLength(c, True)
            circ = float(np.clip(4 * np.pi * area / (peri * peri + 1e-9), 0, 1))
            # Prefer objects not glued to the border and somewhat compact
            border = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
            score = area * (0.25 + 0.75 * circ) * (0.55 if border else 1.0)
            # Prefer lower-half / centre-ish (typical pendulum)
            cy = y + h / 2
            score *= 1.0 + 0.25 * (1.0 - abs(cy - 0.55 * H) / H)
            if circ < 0.2:
                score *= 0.35  # large amorphous motion regions are weaker candidates
            candidates.append(dict(
                x=int(x), y=int(y), w=int(w), h=int(h),
                cx=float(x + w / 2), cy=float(y + h / 2),
                area=float(area), circularity=circ, score=float(score),
                source="motion", border=border,
            ))

    # Colour/saturated blob candidates (works for classic blue ball and other saturated colours)
    hsv0 = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2HSV)
    # Broad saturated mask (any reasonably coloured object)
    sat_mask = cv2.inRange(hsv0, np.array([0, 70, 40], np.uint8), np.array([179, 255, 255], np.uint8))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 60 or area > 0.2 * W * H:
            continue
        x, y, w, h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        circ = float(np.clip(4 * np.pi * area / (peri * peri + 1e-9), 0, 1))
        if circ < 0.25:
            continue
        border = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
        # Strongly prefer round, saturated objects (classic bob) over large motion regions
        score = area * (0.25 + 0.75 * circ) * (0.5 if border else 1.0) * (1.0 + 1.5 * circ)
        candidates.append(dict(
            x=int(x), y=int(y), w=int(w), h=int(h),
            cx=float(x + w / 2), cy=float(y + h / 2),
            area=float(area), circularity=circ, score=float(score),
            source="colour", border=border,
        ))

    # Deduplicate overlapping boxes (keep higher score)
    candidates.sort(key=lambda c: -c["score"])
    kept = []
    for c in candidates:
        overlap = False
        for k in kept:
            # IoU
            xa = max(c["x"], k["x"])
            ya = max(c["y"], k["y"])
            xb = min(c["x"] + c["w"], k["x"] + k["w"])
            yb = min(c["y"] + c["h"], k["y"] + k["h"])
            inter = max(0, xb - xa) * max(0, yb - ya)
            union = c["w"] * c["h"] + k["w"] * k["h"] - inter + 1e-9
            if inter / union > 0.45:
                overlap = True
                break
        if not overlap:
            kept.append(c)
        if len(kept) >= max_candidates:
            break

    # If still nothing, take a few largest mid-frame blobs from edges
    if not kept:
        gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rough = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 100 or area > 0.15 * W * H:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 8 or h < 8:
                continue
            border = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
            if border:
                continue
            rough.append(dict(
                x=int(x), y=int(y), w=int(w), h=int(h),
                cx=float(x + w / 2), cy=float(y + h / 2),
                area=float(area), circularity=0.5, score=float(area),
                source="edge", border=False,
            ))
        rough.sort(key=lambda c: -c["score"])
        kept = rough[:max_candidates]

    best_idx = 0 if kept else -1

    # Annotate first frame for the UI
    annotated = first_bgr.copy()
    for i, c in enumerate(kept):
        color = (40, 220, 90) if i == best_idx else (80, 160, 255)
        cv2.rectangle(annotated, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), color, 2)
        label = "Detected bob" if i == best_idx else f"Candidate {i + 1}"
        cv2.putText(annotated, label, (c["x"], max(18, c["y"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Encode JPEG
    ok_enc, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok_enc:
        raise ValueError("Could not encode preview frame.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return dict(
        width=W, height=H, fps=fps, frame_count=frame_count,
        best_index=best_idx,
        candidates=[{
            "x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"],
            "cx": round(c["cx"], 1), "cy": round(c["cy"], 1),
            "score": round(c["score"], 1), "source": c["source"],
            "circularity": round(c["circularity"], 3),
        } for c in kept],
        preview_jpeg_base64=b64,
    )


def track(path, dark_thr=110, progress=None, prefer="auto", hsv_limits=None,
          ball_click=None, stabilize=True, bob_bbox=None, shape_relax=1.0):
    """Track a bob of arbitrary colour/shape with colour calibration and camera compensation.

    `ball_click` or `bob_bbox` (x,y,w,h in original frame coords) select the object whose
    appearance is learned. `shape_relax` > 1 relaxes circularity filters for non-round bobs.
    Returns raw, camera-stabilized, and zoom-normalized point arrays, apparent size,
    video metadata, and calibration diagnostics.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W <= 0 or H <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("The uploaded video has no readable frames.")
    detection_scale = min(1.0, 720.0 / W) if W > 720 else 1.0
    defaults = dict(DEFAULT_HSV)
    if hsv_limits:
        defaults.update({key: int(value) for key, value in hsv_limits.items() if key in defaults})
    defaults["h_min"] = int(np.clip(defaults["h_min"], 0, 179))
    defaults["h_max"] = int(np.clip(defaults["h_max"], 0, 179))
    defaults["s_min"] = int(np.clip(defaults["s_min"], 0, 255))
    defaults["v_min"] = int(np.clip(defaults["v_min"], 0, 255))
    if defaults["h_max"] <= defaults["h_min"] and not defaults.get("hue_wrap"):
        # Allow h_min > h_max only when hue_wrap is set (red wrap-around)
        if defaults["h_max"] == defaults["h_min"]:
            cap.release()
            raise ValueError("Hue maximum must be greater than hue minimum.")

    shape_relax = max(float(shape_relax or 1.0), 1.0)
    if bob_bbox is not None:
        sample = _sample_bbox_color(cap, bob_bbox, W, H, detection_scale)
        calibration_source = "bbox"
        # User-selected region: allow non-round shapes
        shape_relax = max(shape_relax, 1.8)
    elif ball_click is not None:
        sample = _click_color(cap, ball_click, W, H, detection_scale, defaults)
        calibration_source = "click"
        shape_relax = max(shape_relax, 1.5)
    else:
        sample, best_rank = None, -np.inf
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        calibration_frames = min(frame_count, max(1, int(round(fps * 2.0))))
        sample_step = max(1, int(round(fps / 8.0)))
        for frame_index in range(calibration_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_step:
                continue
            if detection_scale < 1:
                frame = cv2.resize(frame, None, fx=detection_scale, fy=detection_scale,
                                   interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            candidates, _, _ = _blob_candidates(hsv, defaults, detection_scale, shape_relax=shape_relax)
            for candidate in candidates:
                h_med, s_med, _v_med = candidate["hsv"]
                rank = candidate["area"] * (
                    0.48 * candidate["circularity"] + 0.27 * candidate["solidity"]
                    + 0.25 * (s_med / 255.0)
                ) * (0.72 if candidate["border"] else 1.0)
                if rank > best_rank:
                    best_rank, sample = rank, candidate["hsv"]
        calibration_source = "auto" if sample is not None else "defaults"
    limits = _adaptive_hsv(sample, defaults) if sample is not None else defaults
    if progress:
        progress("Calibrated bob colour · " + calibration_source, 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    raw_pos = np.full((frame_count, 2), np.nan, np.float64)
    stable_pos = np.full((frame_count, 2), np.nan, np.float64)
    zoom_pos = np.full((frame_count, 2), np.nan, np.float64)
    size = np.full(frame_count, np.nan, np.float64)
    world_to_frame = np.eye(3, dtype=np.float64)
    prev_gray, prev_feature_mask = None, None
    previous_stable, previous_radius = None, None
    velocity = np.zeros(2, dtype=np.float64)
    missed = 0
    reference_radius = None
    camera_estimates = 0
    actual_n = 0
    sample_indices = set(np.linspace(0, frame_count - 1, min(12, frame_count)).astype(int).tolist())
    debug_samples = []
    max_diagnostics, max_debug_frame = -1, None
    max_candidates, max_rejections, max_debug_choice, debug_index = [], [], None, 0

    for index in range(frame_count):
        ok, original_frame = cap.read()
        if not ok:
            break
        actual_n += 1
        frame = original_frame
        if detection_scale < 1:
            frame = cv2.resize(frame, None, fx=detection_scale, fy=detection_scale,
                               interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        accepted, rejected, blue_mask = _blob_candidates(hsv, limits, detection_scale, shape_relax=shape_relax)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_mask = _background_feature_mask(hsv, blue_mask)

        if prev_gray is not None:
            affine, motion_ok = _estimate_camera_affine(
                prev_gray, gray, prev_feature_mask, feature_mask
            )
            if motion_ok:
                camera_estimates += 1
            affine_h = np.vstack([affine, [0.0, 0.0, 1.0]])
            world_to_frame = affine_h @ world_to_frame
        prev_gray, prev_feature_mask = gray, feature_mask
        camera_scale = float(np.sqrt(max(1e-12, abs(np.linalg.det(world_to_frame[:2, :2])))))
        try:
            frame_to_world = np.linalg.inv(world_to_frame)
        except np.linalg.LinAlgError:
            frame_to_world = np.eye(3, dtype=np.float64)
            world_to_frame = np.eye(3, dtype=np.float64)
            camera_scale = 1.0

        gap = missed + 1
        comparable_previous = previous_stable if previous_stable is not None and missed <= MAX_INTERPOLATED_MISSING_FRAMES else None
        chosen, _distance = _choose_candidate(
            accepted, comparable_previous, velocity, previous_radius, gap,
            frame.shape[:2], prefer, world_to_frame, camera_scale, 1.0, previous_stable,
        )
        tier = 0
        if chosen is None:
            # Tier 2: relaxed shape rules + wider motion window (blur / zoom / partial occlusion)
            acc2, rej2, _m = _blob_candidates(hsv, limits, detection_scale, relaxed=True, shape_relax=shape_relax)
            chosen, _distance = _choose_candidate(
                acc2, comparable_previous, velocity, previous_radius, gap,
                frame.shape[:2], prefer, world_to_frame, camera_scale, 2.2, previous_stable,
            )
            if chosen is not None:
                accepted, rejected, tier = acc2, rej2, 1
        if chosen is None:
            # Tier 3: also widen the colour window (focus / exposure change washes out the blue)
            wide = dict(limits)
            if limits.get("hue_wrap") or limits["h_min"] > limits["h_max"]:
                wide["h_min"] = (limits["h_min"] - 10) % 180
                wide["h_max"] = (limits["h_max"] + 10) % 180
                wide["hue_wrap"] = True
            else:
                wide["h_min"] = max(0, limits["h_min"] - 10)
                wide["h_max"] = min(179, limits["h_max"] + 10)
            wide["s_min"] = max(20, int(limits["s_min"] * 0.5))
            wide["v_min"] = max(10, int(limits["v_min"] * 0.6))
            acc3, rej3, _m = _blob_candidates(hsv, wide, detection_scale, relaxed=True, shape_relax=shape_relax)
            chosen, _distance = _choose_candidate(
                acc3, comparable_previous, velocity, previous_radius, gap,
                frame.shape[:2], prefer, world_to_frame, camera_scale, 3.0, previous_stable,
            )
            if chosen is not None:
                accepted, rejected, tier = acc3, rej3, 2
        if chosen is not None:
            center_work = np.array([chosen["x"], chosen["y"]], dtype=np.float64)
            center_world_work = _transform_point(frame_to_world, center_work)
            center_raw = center_work / detection_scale
            center_stable = center_world_work / detection_scale
            radius_stable = chosen["radius"] / max(camera_scale, 1e-9)
            raw_pos[index], stable_pos[index] = center_raw, center_stable
            size[index] = np.sqrt(chosen["area"]) / detection_scale
            if reference_radius is None:
                reference_radius = radius_stable
            # The camera affine (rotation + UNIFORM SCALE + shift) already removes zoom, so the
            # stabilized point is zoom-normalized.  Re-scaling by the noisy ball radius (old code)
            # created big jumps when the focus/zoom changed, so it is no longer applied.
            zoom_pos[index] = center_stable
            if previous_stable is not None and missed <= MAX_INTERPOLATED_MISSING_FRAMES:
                measured_velocity = (center_world_work - previous_stable) / max(1, gap)
                velocity = 0.65 * velocity + 0.35 * measured_velocity
            else:
                velocity[:] = 0
            previous_stable = center_world_work
            if previous_radius is None or tier == 0:
                previous_radius = radius_stable
            else:   # blurred detections: change the expected size only slowly
                previous_radius = 0.8 * previous_radius + 0.2 * radius_stable
            missed = 0
            for candidate in accepted:
                if candidate is not chosen and "tracking_reject" not in candidate:
                    candidate["tracking_reject"] = "lower colour/shape/prediction score"
        else:
            missed += 1
            for candidate in accepted:
                if "tracking_reject" not in candidate:
                    candidate["tracking_reject"] = "not selected by temporal score"

        diagnostics = len(accepted) + len(rejected)
        if diagnostics > max_diagnostics:
            max_diagnostics = diagnostics
            max_debug_frame = original_frame.copy()
            max_candidates, max_rejections = accepted, rejected
            max_debug_choice = chosen
            debug_index = index
        if index in sample_indices:
            debug_samples.append(dict(
                index=index, image=frame.copy(), candidates=accepted, rejected=rejected,
                chosen=chosen,
            ))
        if progress and index % 12 == 0:
            progress(f"Tracking bob · frame {index + 1} / {frame_count}",
                     6 + 79 * index / max(1, frame_count))
    cap.release()
    if actual_n < frame_count:
        raw_pos, stable_pos, zoom_pos, size = (
            array[:actual_n] for array in (raw_pos, stable_pos, zoom_pos, size)
        )
    if not actual_n or not np.isfinite(raw_pos[:, 0]).any():
        raise ValueError("Bob not found. Confirm the bob in the selection dialog or use click calibration.")

    if max_debug_frame is not None:
        _draw_path(max_debug_frame, raw_pos, 1.0)
        raw_point = raw_pos[debug_index] if debug_index < len(raw_pos) and np.isfinite(raw_pos[debug_index]).all() else None
        _annotate_debug_frame(
            max_debug_frame, max_candidates, max_rejections, max_debug_choice, raw_point, detection_scale
        )
        cv2.imwrite(os.path.splitext(path)[0] + "_debug.jpg", max_debug_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    _write_debug_samples(
        os.path.splitext(path)[0] + "_debug_samples.jpg",
        debug_samples, raw_pos, detection_scale,
    )
    if progress:
        progress("Camera stabilization and debug samples complete", 88)
    info = dict(
        calibration_source=calibration_source,
        calibration_hsv=[round(float(v), 1) for v in sample] if sample is not None else None,
        hsv_limits=limits,
        stabilization_enabled=bool(stabilize),
        zoom_reference_radius=(round(float(reference_radius / detection_scale), 3)
                               if reference_radius is not None else None),
        camera_motion_estimate_rate=(camera_estimates / max(1, actual_n - 1)),
        detection_scale=detection_scale,
    )
    return raw_pos, stable_pos, zoom_pos, size, fps, W, H, info


def clean(pos, size, fps, start=None, end=None, max_gap=None):
    """Interpolate short gaps, pick longest valid run (or user start/end)."""
    n = len(pos)
    idx = np.where(np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1]))[0]
    if len(idx) < 30:
        raise ValueError("Bob not found in enough frames. Confirm the bob selection or adjust colour settings.")
    if max_gap is None:
        max_gap = max(MAX_INTERPOLATED_MISSING_FRAMES + 1, int(round(MAX_ANALYSIS_GAP_SECONDS * fps)))
    allf = np.arange(n)
    for a in (pos[:, 0], pos[:, 1], size):
        a[:] = np.interp(allf, idx, a[idx], left=np.nan, right=np.nan)
    gaps = np.diff(idx)
    for a, b in zip(idx[:-1][gaps > max_gap], idx[1:][gaps > max_gap]):
        pos[a + 1:b] = np.nan
        size[a + 1:b] = np.nan
    ok = ~np.isnan(pos[:, 0])
    if start is not None or end is not None:
        i0, i1 = int((start or 0) * fps), int((end or n / fps) * fps)
    else:
        i0 = i1 = 0
        st = None
        for i, v in enumerate(np.append(ok, False)):
            if v and st is None:
                st = i
            elif not v and st is not None:
                if i - st > i1 - i0:
                    i0, i1 = st, i
                st = None
    out = np.zeros(n, bool)
    out[i0:i1] = True
    pos[~out] = np.nan
    size[~out] = np.nan
    return i0, min(i1, n)


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


def _form_bool(form, name, default=True):
    if name not in form:
        return default
    return str(form.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _aligned_coordinates(source, start, end, max_gap=None):
    """Interpolate only short tracking gaps and keep exports aligned to the analysed run."""
    source = np.asarray(source, dtype=np.float64)
    out = np.full_like(source, np.nan)
    valid = np.where(np.isfinite(source[:, 0]) & np.isfinite(source[:, 1]))[0]
    if not len(valid):
        return out
    frames = np.arange(len(source))
    for axis in range(2):
        out[:, axis] = np.interp(frames, valid, source[valid, axis], left=np.nan, right=np.nan)
    if max_gap is None:
        max_gap = MAX_INTERPOLATED_MISSING_FRAMES + 1
    gaps = np.diff(valid)
    for left, right in zip(valid[:-1][gaps > max_gap], valid[1:][gaps > max_gap]):
        out[left + 1:right] = np.nan
    out[:start] = np.nan
    out[end:] = np.nan
    return out


def run_job(jid, path, form):
    """Runs in a worker thread; updates JOBS[jid]['pct'/'stage'] so the browser can show real progress."""
    job = JOBS[jid]

    def progress(stage, pct):
        job["stage"], job["pct"] = stage, round(float(pct), 1)

    try:
        scale = float(form.get("scale") or 0)
        dark_thr = int(form.get("dark_thr") or 110)
        prefer = (form.get("prefer") or "auto").strip().lower()
        hsv_limits = {
            key: int(form.get(key)) if form.get(key) not in (None, "") else default
            for key, default in DEFAULT_HSV.items()
        }
        camera_stabilize = _form_bool(form, "camera_stabilize", True)
        zoom_normalize = _form_bool(form, "zoom_normalize", True)
        click_x, click_y = form.get("ball_click_x"), form.get("ball_click_y")
        ball_click = (float(click_x), float(click_y)) if click_x not in (None, "") and click_y not in (None, "") else None
        bob_bbox = None
        bx = form.get("bob_bbox_x")
        if bx not in (None, ""):
            try:
                bob_bbox = (
                    float(form.get("bob_bbox_x")),
                    float(form.get("bob_bbox_y")),
                    float(form.get("bob_bbox_w")),
                    float(form.get("bob_bbox_h")),
                )
            except (TypeError, ValueError):
                bob_bbox = None
        shape_relax = float(form.get("shape_relax") or 1.0)
        mass = form.get("mass") or ""
        length = form.get("length") or ""
        g_val = form.get("g") or "9.81"
        trial = (form.get("trial") or "").strip()
        notes = (form.get("notes") or "").strip()
        try:
            mass_f = float(mass) if mass != "" else None
        except ValueError:
            mass_f = None
        try:
            length_f = float(length) if length != "" else None
        except ValueError:
            length_f = None
        try:
            g_f = float(g_val) if g_val != "" else 9.81
        except ValueError:
            g_f = 9.81

        raw_pos, stable_pos, zoom_pos, size, fps, W, H, tracking = track(
            path, dark_thr, progress, prefer, hsv_limits, ball_click, camera_stabilize,
            bob_bbox=bob_bbox, shape_relax=shape_relax,
        )
        if camera_stabilize:
            pos = zoom_pos.copy() if zoom_normalize else stable_pos.copy()
        else:
            pos = raw_pos.copy()
        size = size.copy()
        measured = np.isfinite(raw_pos[:, 0])          # frames where the ball was really detected
        start = float(form["start"]) if form.get("start") else None
        end = float(form["end"]) if form.get("end") else None
        progress("Cleaning the trajectory", 93)
        gap_limit = max(MAX_INTERPOLATED_MISSING_FRAMES + 1, int(round(MAX_ANALYSIS_GAP_SECONDS * fps)))
        i0, i1 = clean(pos, size, fps, start, end, gap_limit)
        raw_coordinates = _aligned_coordinates(raw_pos, i0, i1, gap_limit)
        stabilized_coordinates = _aligned_coordinates(stable_pos, i0, i1, gap_limit)
        zoom_coordinates = _aligned_coordinates(zoom_pos, i0, i1, gap_limit)
        coverage = float(measured[i0:i1].mean()) if i1 > i0 else float("nan")
        n = len(pos)
        t = np.arange(n) / fps
        s = scale or 1.0
        mx, my = np.nanmean(pos[:, 0]), np.nanmean(pos[:, 1])
        x, y = (pos[:, 0] - mx) * s, -(pos[:, 1] - my) * s
        ts, xs, ys = t[i0:i1] - t[i0], _fill_nan(x[i0:i1]), _fill_nan(y[i0:i1])
        progress("Measuring periods, amplitudes and shape", 96)
        Tx, Txs, nx = peak_period(ts, xs, fps)
        Ty, Tys, ny = peak_period(ts, ys, fps)
        fxf, fyf = fft_peak(xs, fps), fft_peak(ys, fps)
        Ax, Ay = amplitude(xs), amplitude(ys)
        phase = phase_diff_deg(xs, ys, fps)
        decay = amplitude_decay(ts, xs, ys)
        At_t, At_v, gamma, A0 = envelope_and_damping(ts, xs, ys, fps)
        # whole-segment PCA for eccentricity / mean θ
        th0, ecc0 = pca(xs[np.isfinite(xs) & np.isfinite(ys)], ys[np.isfinite(xs) & np.isfinite(ys)])
        progress("Fitting the trajectory axis θ(t)", 98)
        rows, th, rate = sliding(t[i0:i1], xs, ys, fps)
        try:
            drift = float(np.polyfit(rows[:, 0], np.degrees(np.unwrap(np.radians(2 * rows[:, 3])) / 2), 1)[0])
        except Exception:
            drift = float("nan")
        fx_peak = (1.0 / Tx) if np.isfinite(Tx) and Tx != 0 else float("nan")
        fy_peak = (1.0 / Ty) if np.isfinite(Ty) and Ty != 0 else float("nan")
        df_peak = fx_peak - fy_peak if np.isfinite(fx_peak) and np.isfinite(fy_peak) else float("nan")
        df_fft = (fxf - fyf) if np.isfinite(fxf) and np.isfinite(fyf) else float("nan")
        freq_ratio = (fx_peak / fy_peak) if np.isfinite(fx_peak) and np.isfinite(fy_peak) and fy_peak != 0 else float("nan")
        freq_ratio_fft = (fxf / fyf) if np.isfinite(fxf) and np.isfinite(fyf) and fyf != 0 else float("nan")
        res = (1.0 / ts[-1]) if len(ts) and ts[-1] != 0 else float("nan")

        # theoretical period from L, g
        T_theory = float("nan")
        if length_f is not None and length_f > 0 and g_f > 0:
            L_m = length_f / 100.0 if length_f > 3 else length_f
            T_theory = 2.0 * np.pi * np.sqrt(L_m / g_f)
        T_meas = nanmean2(Tx, Ty)
        T_err = (T_meas - T_theory) / T_theory if np.isfinite(T_meas) and np.isfinite(T_theory) and T_theory else float("nan")

        shape_label, shape_detail = classify_shape(Ax, Ay, ecc0, phase, df_peak, drift, decay)
        circularity = float(ecc0) if np.isfinite(ecc0) else float("nan")

        params = dict(
            dark_thr=dark_thr,
            h_min=hsv_limits["h_min"], h_max=hsv_limits["h_max"],
            s_min=hsv_limits["s_min"], v_min=hsv_limits["v_min"],
            calibrated_hsv=tracking["calibration_hsv"],
            calibration_source=tracking["calibration_source"],
            active_hsv=tracking["hsv_limits"],
            camera_stabilize=camera_stabilize,
            zoom_normalize=zoom_normalize,
            camera_motion_estimate_rate=tracking["camera_motion_estimate_rate"],
            scale=scale or None,
            unit="cm" if scale else "px",
            start=start, end=end,
            mass=mass_f, length=length_f, g=g_f,
            trial=trial or None, notes=notes or None,
        )
        summary = dict(
            Tx=Tx, Tx_sd=Txs, Ty=Ty, Ty_sd=Tys,
            fx_peak=fx_peak, fy_peak=fy_peak,
            fx_fft=fxf, fy_fft=fyf,
            df_peak=df_peak, df_fft=df_fft,
            freq_ratio=freq_ratio, freq_ratio_fft=freq_ratio_fft,
            Ax=Ax, Ay=Ay, A0=A0, gamma=gamma,
            amplitude_decay=decay,
            eccentricity=ecc0, circularity=circularity,
            phase_deg=phase,  # Δφ
            theta_mean=th0, theta_drift=drift,
            shape=shape_label, shape_detail=shape_detail,
            T_theory=T_theory, T_meas=T_meas, T_rel_error=T_err,
            resolution=res, tracking_coverage=coverage,
            t_start=float(t[i0]), t_end=float(t[i1 - 1]),
            z_note="z(t) estimated from bob apparent size (single camera); true 3-D needs synchronized side view",
        )
        med = np.nanmedian(size) if np.isfinite(size).any() else 1.0
        if not np.isfinite(med) or med == 0:
            med = 1.0
        R = float(np.nanmax(np.abs(np.r_[x, y]))) if np.isfinite(np.r_[x, y]).any() else 1.0
        if not np.isfinite(R) or R == 0:
            R = 1.0
        z = size / med - 1
        # map envelope onto full timeline (NaN outside analysed segment)
        A_full = np.full(n, np.nan)
        if len(At_t):
            # At_t is relative to segment start; convert to absolute time
            At_abs = At_t + float(t[i0])
            for ti, ai in zip(At_abs, At_v):
                idx = int(round(ti * fps))
                if 0 <= idx < n:
                    A_full[idx] = ai
            # interpolate sparse envelope for export continuity inside segment
            ok = np.isfinite(A_full)
            if ok.sum() >= 2:
                A_full[i0:i1] = np.interp(
                    t[i0:i1], t[ok], A_full[ok], left=np.nan, right=np.nan
                )
        job.update(
            path=path, t=t, x=x, y=y, z=z, A=A_full, unit="cm" if scale else "px",
            params=params, summary=summary, rows_raw=(rows, th, rate),
            pos=pos, raw_coordinates=raw_coordinates,
            stabilized_coordinates=stabilized_coordinates,
            zoom_coordinates=zoom_coordinates, i0=i0, i1=i1, fps=fps,
            At_t=At_t + float(t[i0]) if len(At_t) else At_t, At_v=At_v,
        )
        job["result"] = dict(
            id=jid, fps=fps, n=n, W=W, H=H, unit="cm" if scale else "px",
            # px/py are consumed by the existing video overlay and must stay in frame pixels.
            px=L(raw_coordinates[:, 0], 1), py=L(raw_coordinates[:, 1], 1),
            x=L(x), y=L(y), z=L(z),
            sz=L(size, 1), meas=[int(bool(v)) for v in measured],
            raw_x=L(raw_coordinates[:, 0], 1), raw_y=L(raw_coordinates[:, 1], 1),
            stabilized_x=L(stabilized_coordinates[:, 0], 1),
            stabilized_y=L(stabilized_coordinates[:, 1], 1),
            zoom_normalized_x=L(zoom_coordinates[:, 0], 1),
            zoom_normalized_y=L(zoom_coordinates[:, 1], 1),
            A=L(A_full),  # A(t) envelope
            R=R,
            params={k: (safe(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in params.items()},
            summary={k: (v if isinstance(v, str) else safe(v)) for k, v in summary.items()},
            rows=[dict(t=round(float(r_[0]), 3), Tx=L([r_[1]])[0], Ty=L([r_[2]])[0], th=L([a])[0], ecc=L([r_[4]])[0],
                       rate=L([b])[0]) for r_, a, b in zip(rows, th, rate)],
            envelope=[dict(t=round(float(ti), 3), A=L([ai])[0]) for ti, ai in zip(
                (At_t + float(t[i0]) if len(At_t) else At_t), At_v)])
        progress("Done", 100)
        job["state"] = "done"
    except Exception as e:  # noqa
        job["state"], job["error"] = "error", str(e) or e.__class__.__name__


@app.route("/detect_bob", methods=["POST"])
def detect_bob():
    """Analyse the first frames of an uploaded video and return candidate bob boxes."""
    f = request.files.get("video")
    if not f:
        return jsonify(error="No file received"), 400
    tmp_name = uuid.uuid4().hex[:12] + (os.path.splitext(f.filename or "")[1] or ".mp4")
    path = os.path.join(UPLOADS, tmp_name)
    try:
        f.save(path)
        result = detect_bob_candidates(path)
        return jsonify(result)
    except Exception as e:  # noqa
        return jsonify(error=str(e) or e.__class__.__name__), 400
    finally:
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f:
        return jsonify(error="No file received")
    jid = uuid.uuid4().hex[:10]
    path = os.path.join(UPLOADS, jid + (os.path.splitext(f.filename or "")[1] or ".mp4"))
    f.save(path)
    JOBS[jid] = dict(path=path, state="running", pct=0.0, stage="Starting", name=f.filename)
    threading.Thread(target=run_job, args=(jid, path, request.form.to_dict()), daemon=True).start()
    return jsonify(job=jid)


@app.route("/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j:
        return jsonify(state="error", error="Unknown job (the server was restarted?)"), 404
    return jsonify(state=j["state"], pct=j["pct"], stage=j["stage"], error=j.get("error"))


@app.route("/result/<jid>")
def result(jid):
    j = JOBS.get(jid)
    if not j or "result" not in j:
        return jsonify(error="Result not ready"), 404
    return jsonify(j["result"])


@app.route("/video/<jid>")
def video(jid):
    if jid not in JOBS:
        abort(404)
    p = JOBS[jid]["path"]
    return send_from_directory(os.path.dirname(p), os.path.basename(p), conditional=True)


@app.route("/debug/<jid>")
def debug_img(jid):
    """Tracking diagnostic with the selected ball, rejected components, and path."""
    if jid not in JOBS:
        abort(404)
    p = os.path.splitext(JOBS[jid]["path"])[0] + "_debug.jpg"
    if not os.path.exists(p):
        abort(404)
    return send_from_directory(os.path.dirname(p), os.path.basename(p))


@app.route("/debug-samples/<jid>")
def debug_samples(jid):
    """Contact sheet of sampled tracking decisions from across the video."""
    if jid not in JOBS:
        abort(404)
    p = os.path.splitext(JOBS[jid]["path"])[0] + "_debug_samples.jpg"
    if not os.path.exists(p):
        abort(404)
    return send_from_directory(os.path.dirname(p), os.path.basename(p))


@app.route("/csv/<jid>")
def csv_dl(jid):
    """Full export: parameters + summary header, then full time-series table."""
    if jid not in JOBS or "x" not in JOBS[jid]:
        abort(404)
    j = JOBS[jid]
    unit = j.get("unit", "px")
    params = j.get("params") or {}
    summary = j.get("summary") or {}
    buf = io.StringIO()
    w = csv.writer(buf)

    w.writerow(["# Object Trajectory Analysis — full export"])
    w.writerow(["# --- parameters ---"])
    for k in ("trial", "notes", "h_min", "h_max", "s_min", "v_min",
              "calibrated_hsv", "calibration_source", "active_hsv",
              "camera_stabilize", "zoom_normalize", "camera_motion_estimate_rate",
              "dark_thr", "scale", "unit", "start", "end", "mass", "length", "g"):
        v = params.get(k, "")
        w.writerow([f"# param_{k}", "" if v is None else v])
    w.writerow(["# --- summary ---"])
    for k, v in summary.items():
        if isinstance(v, float) and not np.isfinite(v):
            v = ""
        w.writerow([f"# summary_{k}", v])
    w.writerow([])
    w.writerow([
        "time_s", f"x_{unit}", f"y_{unit}",
        "raw_x_px", "raw_y_px", "stabilized_x_px", "stabilized_y_px",
        "zoom_normalized_x_px", "zoom_normalized_y_px",
        "z_est", "r", "phi_deg", f"A_envelope_{unit}",
    ])
    t, x, y = j["t"], j["x"], j["y"]
    raw_coordinates = j.get("raw_coordinates")
    stabilized_coordinates = j.get("stabilized_coordinates")
    zoom_coordinates = j.get("zoom_coordinates")
    z = j.get("z")
    A = j.get("A")
    for i, (a, b, c) in enumerate(zip(t, x, y)):
        if not np.isfinite(b):
            continue
        raw_x, raw_y = raw_coordinates[i] if raw_coordinates is not None else (np.nan, np.nan)
        stable_x, stable_y = stabilized_coordinates[i] if stabilized_coordinates is not None else (np.nan, np.nan)
        zoom_x, zoom_y = zoom_coordinates[i] if zoom_coordinates is not None else (np.nan, np.nan)
        zi = z[i] if z is not None and i < len(z) and np.isfinite(z[i]) else ""
        Ai = A[i] if A is not None and i < len(A) and np.isfinite(A[i]) else ""
        r = float(np.hypot(b, c)) if np.isfinite(c) else ""
        phi = float(np.degrees(np.arctan2(c, b))) if np.isfinite(c) else ""
        w.writerow([
                    f"{a:.4f}", f"{b:.4f}", f"{c:.4f}" if np.isfinite(c) else "",
                    f"{raw_x:.3f}" if np.isfinite(raw_x) else "",
                    f"{raw_y:.3f}" if np.isfinite(raw_y) else "",
                    f"{stable_x:.3f}" if np.isfinite(stable_x) else "",
                    f"{stable_y:.3f}" if np.isfinite(stable_y) else "",
                    f"{zoom_x:.3f}" if np.isfinite(zoom_x) else "",
                    f"{zoom_y:.3f}" if np.isfinite(zoom_y) else "",
                    f"{zi:.4f}" if zi != "" else "",
                    f"{r:.4f}" if r != "" else "",
                    f"{phi:.3f}" if phi != "" else "",
                    f"{Ai:.4f}" if Ai != "" else ""])
    # sliding window table
    rows_raw = j.get("rows_raw")
    if rows_raw is not None:
        rows, th, rate = rows_raw
        w.writerow([])
        w.writerow(["# sliding window (θ, periods) — trajectory evolution/rotation"])
        w.writerow(["t_s", "Tx_s", "Ty_s", "theta_deg", "ecc_minor_over_major", "theta_rate_deg_per_s"])
        for r_, a, b in zip(rows, th, rate):
            w.writerow([
                f"{r_[0]:.3f}",
                f"{r_[1]:.4f}" if np.isfinite(r_[1]) else "",
                f"{r_[2]:.4f}" if np.isfinite(r_[2]) else "",
                f"{a:.3f}" if np.isfinite(a) else "",
                f"{r_[4]:.4f}" if np.isfinite(r_[4]) else "",
                f"{b:.3f}" if np.isfinite(b) else "",
            ])
    # sparse envelope samples
    At_t, At_v = j.get("At_t"), j.get("At_v")
    if At_t is not None and len(At_t):
        w.writerow([])
        w.writerow(["# amplitude envelope A(t) samples"])
        w.writerow(["t_s", f"A_{unit}"])
        for ti, ai in zip(At_t, At_v):
            if np.isfinite(ai):
                w.writerow([f"{ti:.3f}", f"{ai:.4f}"])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=trajectory_full.csv"})


@app.route("/json/<jid>")
def json_dl(jid):
    """Full JSON export: parameters, summary, time series, sliding windows, A(t)."""
    j = JOBS.get(jid)
    if not j or "result" not in j:
        abort(404)
    r = j["result"]
    payload = dict(
        id=r["id"], fps=r["fps"], n=r["n"], W=r["W"], H=r["H"], unit=r["unit"],
        params=r.get("params"), summary=r.get("summary"),
        time_series=dict(
            t=L(j["t"], 4), x=L(j["x"]), y=L(j["y"]),
            raw_x=r.get("raw_x"), raw_y=r.get("raw_y"),
            stabilized_x=r.get("stabilized_x"), stabilized_y=r.get("stabilized_y"),
            zoom_normalized_x=r.get("zoom_normalized_x"),
            zoom_normalized_y=r.get("zoom_normalized_y"),
            z=L(j.get("z", np.array([]))),
            A=L(j.get("A", np.array([]))),
        ),
        sliding=r.get("rows"),
        envelope=r.get("envelope"),
    )
    return Response(
        __import__("json").dumps(payload, allow_nan=False),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=trajectory_full.json"},
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), threaded=True)
