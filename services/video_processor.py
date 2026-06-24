import os
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try: 
    import cv2
    CV2_AVAILABLE = True
except:
    cv2= None
    CV2_AVAILABLE = False

try:
    from mmpose.apis import MMPoseInferencer
    MMPOSE_AVAILABLE = True
    MMPOSE_IMPORT_ERROR = None
except Exception as e:
    MMPoseInferencer = None
    MMPOSE_AVAILABLE = False
    MMPOSE_IMPORT_ERROR = repr(e)

from config import (
    TARGET_FRAMES,
    SCORE_THRESHOLD,
    AUX_DIM,
    AUX_MEAN_PATH,
    AUX_STD_PATH,
    KEYPOINTS_DIR,
    RTMPOSE_DEVICE,
    EXTRACT_FRAME_STEP,
    MAX_EXTRACT_FRAMES,
    VIDEO_EXTENSIONS,
    WINDOW_STRIDE,
    USE_MIRROR_TTA,
)

EPS = 1e-6
NUM_WHOLEBODY_KEYPOINTS = 133

# COCO-WholeBody layout used by RTMPose.
BODY_SLICE = slice(0, 17)
FEET_SLICE = slice(17, 23)
FACE_SLICE = slice(23, 91)
LEFT_HAND_SLICE = slice(91, 112)
RIGHT_HAND_SLICE = slice(112, 133)

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST_BODY = 9
RIGHT_WRIST_BODY = 10
LEFT_HIP = 11
RIGHT_HIP = 12

UPPER_SELECTED_INDICES = (
    [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST_BODY, RIGHT_WRIST_BODY]
    + list(range(LEFT_HAND_SLICE.start, LEFT_HAND_SLICE.stop))
    + list(range(RIGHT_HAND_SLICE.start, RIGHT_HAND_SLICE.stop))
)
LEFT_HAND_UPPER_LOCAL_SLICE = slice(7, 28)
RIGHT_HAND_UPPER_LOCAL_SLICE = slice(28, 49)

HAND_EDGES_21 = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

UPPER_EDGES_49 = [
    (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (0, 1), (0, 2),
    (5, 7), (6, 28),
]
for a, b in HAND_EDGES_21:
    UPPER_EDGES_49.append((7 + a, 7 + b))
    UPPER_EDGES_49.append((28 + a, 28 + b))

HAND42_EDGES = []
for a, b in HAND_EDGES_21:
    HAND42_EDGES.append((a, b))
    HAND42_EDGES.append((21 + a, 21 + b))

WHOLEBODY_SWAP = list(range(NUM_WHOLEBODY_KEYPOINTS))
for a, b in [(LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_ELBOW, RIGHT_ELBOW), (LEFT_WRIST_BODY, RIGHT_WRIST_BODY), (LEFT_HIP, RIGHT_HIP)]:
    WHOLEBODY_SWAP[a], WHOLEBODY_SWAP[b] = WHOLEBODY_SWAP[b], WHOLEBODY_SWAP[a]
for i in range(21):
    WHOLEBODY_SWAP[LEFT_HAND_SLICE.start + i], WHOLEBODY_SWAP[RIGHT_HAND_SLICE.start + i] = (
        WHOLEBODY_SWAP[RIGHT_HAND_SLICE.start + i],
        WHOLEBODY_SWAP[LEFT_HAND_SLICE.start + i],
    )
WHOLEBODY_SWAP = np.array(WHOLEBODY_SWAP, dtype=np.int64)

def make_adjacency(num_nodes, edges):
    A = np.eye(num_nodes, dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    return (A / np.maximum(deg, 1e-6)).astype(np.float32)

A_UPPER49 = make_adjacency(49, UPPER_EDGES_49)
A_HAND42 = make_adjacency(42, HAND42_EDGES)

_AUX_MEAN = None
_AUX_STD = None
_RTMPOSE_INFERENCER = None

def load_aux_stats() -> Tuple[np.ndarray, np.ndarray]:
    global _AUX_MEAN, _AUX_STD
    if _AUX_MEAN is None or _AUX_STD is None:
        if not AUX_MEAN_PATH.exists() or not AUX_STD_PATH.exists():
            raise FileNotFoundError(
                f"Missing aux stats. Expected: {AUX_MEAN_PATH} and {AUX_STD_PATH}"
            )
        _AUX_MEAN = np.load(AUX_MEAN_PATH).astype(np.float32)
        _AUX_STD = np.load(AUX_STD_PATH).astype(np.float32)
        _AUX_STD = np.where(_AUX_STD < 1e-6, 1.0, _AUX_STD).astype(np.float32)
    return _AUX_MEAN, _AUX_STD

def sanitize_numeric(x, clip_value=1e6):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.number):
        return x
    x = x.astype(np.float32)
    bad = ~np.isfinite(x)
    if bad.any():
        x[bad] = 0.0
    x = np.clip(x, -clip_value, clip_value)
    return x.astype(np.float32)

def make_landmark_mask(scores, score_thr=SCORE_THRESHOLD):
    return (scores >= score_thr).astype(np.uint8)

def safe_center(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((2,), dtype=np.float32)
    return np.median(points, axis=0).astype(np.float32)

def safe_bbox_scale(points):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return 1.0
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return float(max(x_max - x_min, y_max - y_min, 1.0))

def valid_point(scores, t, idx, thr=SCORE_THRESHOLD):
    return bool(scores[t, idx] >= thr)

def compute_hand_center_and_scale(raw_hand_points, raw_hand_scores):
    raw_hand_points = np.asarray(raw_hand_points, dtype=np.float32)
    raw_hand_scores = np.asarray(raw_hand_scores, dtype=np.float32)
    visible = raw_hand_scores >= SCORE_THRESHOLD
    if visible.sum() < 3:
        return np.zeros((2,), dtype=np.float32), 1.0, 0, 0.0, np.zeros((4,), dtype=np.float32)
    pts = raw_hand_points[visible]
    center = safe_center(pts)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    scale = float(max(x_max - x_min, y_max - y_min, EPS))
    if scale <= EPS and visible[0] and visible[9]:
        scale = float(np.linalg.norm(raw_hand_points[0] - raw_hand_points[9]))
    if scale <= EPS:
        scale = 1.0
    area = max(float(x_max - x_min), 0.0) * max(float(y_max - y_min), 0.0)
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    return center, scale, 1, area, box

def fill_invalid_vector(values, valid_mask):
    values = values.copy().astype(np.float32)
    valid_mask = valid_mask.astype(bool)
    if valid_mask.sum() == 0:
        return values
    median_value = np.median(values[valid_mask], axis=0)
    values[~valid_mask] = median_value
    return values

def fill_invalid_scalar(values, valid_mask, default=1.0):
    values = values.copy().astype(np.float32)
    valid_mask = valid_mask.astype(bool)
    if valid_mask.sum() == 0:
        values[:] = default
        return values
    median_value = float(np.median(values[valid_mask]))
    if median_value <= EPS:
        median_value = default
    bad = (~valid_mask) | (values <= EPS)
    values[bad] = median_value
    return values

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

def rotation_matrix(theta):
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)

def ensure_64_frames(keypoints, scores=None, frame_mask=None):
    """Uniformly sample or pad any sequence to TARGET_FRAMES.
    Important: for long videos, this samples across the whole video, not the last 64 frames.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    T = keypoints.shape[0]
    if scores is None:
        scores = np.ones((T, keypoints.shape[1]), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)
    if frame_mask is None:
        frame_mask = np.ones((T,), dtype=np.uint8)
    else:
        frame_mask = np.asarray(frame_mask, dtype=np.uint8)

    if T == TARGET_FRAMES:
        return keypoints, scores, frame_mask, np.arange(T)

    if T > TARGET_FRAMES:
        idx = np.linspace(0, T - 1, TARGET_FRAMES).round().astype(np.int64)
        return keypoints[idx], scores[idx], frame_mask[idx], idx

    kp = np.zeros((TARGET_FRAMES, keypoints.shape[1], keypoints.shape[2]), dtype=np.float32)
    sc = np.zeros((TARGET_FRAMES, scores.shape[1]), dtype=np.float32)
    fm = np.zeros((TARGET_FRAMES,), dtype=np.uint8)
    kp[:T] = keypoints
    sc[:T] = scores
    fm[:T] = frame_mask
    return kp, sc, fm, np.arange(T)

def load_raw_keypoints_npz(path):
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    if "keypoints" in data.files:
        keypoints = data["keypoints"].astype(np.float32)
    elif "kpts" in data.files:
        keypoints = data["kpts"].astype(np.float32)
    else:
        raise KeyError(f"No keypoints array found in {path}. Available keys={data.files}")

    if "keypoint_scores" in data.files:
        scores = data["keypoint_scores"].astype(np.float32)
    elif "scores" in data.files:
        scores = data["scores"].astype(np.float32)
    else:
        scores = np.ones((keypoints.shape[0], keypoints.shape[1]), dtype=np.float32)

    if "frame_mask" in data.files:
        frame_mask = data["frame_mask"].astype(np.uint8)
    else:
        frame_mask = np.ones((keypoints.shape[0],), dtype=np.uint8)

    return keypoints, scores, frame_mask, data

def extract_or_create_masks(keypoints, scores, frame_mask):
    landmark_mask = make_landmark_mask(scores)
    body_mask = landmark_mask[:, BODY_SLICE]
    face_mask = landmark_mask[:, FACE_SLICE]
    left_hand_mask = landmark_mask[:, LEFT_HAND_SLICE]
    right_hand_mask = landmark_mask[:, RIGHT_HAND_SLICE]
    landmark_mask = landmark_mask * frame_mask[:, None]
    body_mask = body_mask * frame_mask[:, None]
    face_mask = face_mask * frame_mask[:, None]
    left_hand_mask = left_hand_mask * frame_mask[:, None]
    right_hand_mask = right_hand_mask * frame_mask[:, None]
    return landmark_mask, body_mask, face_mask, left_hand_mask, right_hand_mask

def compute_robust_center_scale_angle(keypoints, scores, landmark_mask, frame_mask):
    centers = np.zeros((TARGET_FRAMES, 2), dtype=np.float32)
    scales = np.ones((TARGET_FRAMES,), dtype=np.float32)
    angles = np.zeros((TARGET_FRAMES,), dtype=np.float32)
    center_valid = np.zeros((TARGET_FRAMES,), dtype=np.uint8)
    scale_valid = np.zeros((TARGET_FRAMES,), dtype=np.uint8)
    angle_valid = np.zeros((TARGET_FRAMES,), dtype=np.uint8)

    for t in range(TARGET_FRAMES):
        if frame_mask[t] == 0:
            continue
        ls_ok = valid_point(scores, t, LEFT_SHOULDER)
        rs_ok = valid_point(scores, t, RIGHT_SHOULDER)
        if ls_ok and rs_ok:
            ls = keypoints[t, LEFT_SHOULDER].astype(np.float32)
            rs = keypoints[t, RIGHT_SHOULDER].astype(np.float32)
            centers[t] = (ls + rs) / 2.0
            center_valid[t] = 1
            shoulder_dist = float(np.linalg.norm(ls - rs))
            if shoulder_dist > EPS:
                scales[t] = shoulder_dist
                scale_valid[t] = 1
            dx, dy = rs - ls
            angles[t] = normalize_angle(math.atan2(float(dy), float(dx)))
            angle_valid[t] = 1
            continue

        upper_body_indices = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST_BODY, RIGHT_WRIST_BODY]
        visible_upper = [keypoints[t, idx] for idx in upper_body_indices if scores[t, idx] >= SCORE_THRESHOLD]
        if len(visible_upper) >= 3:
            pts = np.asarray(visible_upper, dtype=np.float32)
            centers[t] = safe_center(pts)
            scales[t] = safe_bbox_scale(pts)
            center_valid[t] = 1
            scale_valid[t] = 1
            continue

        hand_centers, hand_scales = [], []
        lc, ls, lv, _, _ = compute_hand_center_and_scale(keypoints[t, LEFT_HAND_SLICE], scores[t, LEFT_HAND_SLICE])
        rc, rs, rv, _, _ = compute_hand_center_and_scale(keypoints[t, RIGHT_HAND_SLICE], scores[t, RIGHT_HAND_SLICE])
        if lv:
            hand_centers.append(lc); hand_scales.append(ls)
        if rv:
            hand_centers.append(rc); hand_scales.append(rs)
        if len(hand_centers) > 0:
            centers[t] = safe_center(np.asarray(hand_centers, dtype=np.float32))
            scales[t] = float(np.median(hand_scales)) if hand_scales else 1.0
            center_valid[t] = 1
            scale_valid[t] = 1
            continue

        visible_all = scores[t] >= SCORE_THRESHOLD
        if visible_all.sum() >= 3:
            pts = keypoints[t, visible_all].astype(np.float32)
            centers[t] = safe_center(pts)
            scales[t] = safe_bbox_scale(pts)
            center_valid[t] = 1
            scale_valid[t] = 1

    centers = fill_invalid_vector(centers, center_valid)
    scales = fill_invalid_scalar(scales, scale_valid, default=1.0)
    if angle_valid.sum() > 0:
        median_angle = float(np.median(angles[angle_valid == 1]))
        angles[angle_valid == 0] = median_angle
    else:
        angles[:] = 0.0

    return {
        "centers": centers.astype(np.float32),
        "scales": scales.astype(np.float32),
        "angles": angles.astype(np.float32),
        "center_valid": center_valid.astype(np.uint8),
        "scale_valid": scale_valid.astype(np.uint8),
        "angle_valid": angle_valid.astype(np.uint8),
    }

def body_normalize_keypoints(keypoints, centers, scales, angles, landmark_mask, frame_mask, align_shoulders=True):
    body_norm = np.zeros_like(keypoints, dtype=np.float32)
    for t in range(TARGET_FRAMES):
        if frame_mask[t] == 0:
            continue
        shifted = keypoints[t].astype(np.float32) - centers[t][None, :]
        if align_shoulders:
            R = rotation_matrix(-float(angles[t]))
            shifted = shifted @ R.T
        normed = shifted / (float(scales[t]) + EPS)
        normed[landmark_mask[t] == 0] = 0.0
        body_norm[t] = normed.astype(np.float32)
    return body_norm

def extract_upper49(normalized_keypoints, landmark_mask):
    X_upper = normalized_keypoints[:, UPPER_SELECTED_INDICES, :].astype(np.float32)
    upper_mask = landmark_mask[:, UPPER_SELECTED_INDICES].astype(np.uint8)
    return (X_upper * upper_mask[..., None]).astype(np.float32), upper_mask.astype(np.uint8)

def extract_lr_hands_from_body_norm(normalized_keypoints, landmark_mask):
    left = normalized_keypoints[:, LEFT_HAND_SLICE, :].astype(np.float32)
    right = normalized_keypoints[:, RIGHT_HAND_SLICE, :].astype(np.float32)
    left_mask = landmark_mask[:, LEFT_HAND_SLICE].astype(np.uint8)
    right_mask = landmark_mask[:, RIGHT_HAND_SLICE].astype(np.uint8)
    X = np.concatenate([left, right], axis=1)
    M = np.concatenate([left_mask, right_mask], axis=1)
    return (X * M[..., None]).astype(np.float32), M.astype(np.uint8)

def hand_local_normalize_raw(keypoints, scores, frame_mask):
    X_hand_local = np.zeros((TARGET_FRAMES, 42, 2), dtype=np.float32)
    hand_local_mask = np.zeros((TARGET_FRAMES, 42), dtype=np.uint8)
    left_hand_scales = np.ones((TARGET_FRAMES,), dtype=np.float32)
    right_hand_scales = np.ones((TARGET_FRAMES,), dtype=np.float32)
    left_hand_areas = np.zeros((TARGET_FRAMES,), dtype=np.float32)
    right_hand_areas = np.zeros((TARGET_FRAMES,), dtype=np.float32)
    left_hand_centers = np.zeros((TARGET_FRAMES, 2), dtype=np.float32)
    right_hand_centers = np.zeros((TARGET_FRAMES, 2), dtype=np.float32)
    left_hand_valid = np.zeros((TARGET_FRAMES,), dtype=np.uint8)
    right_hand_valid = np.zeros((TARGET_FRAMES,), dtype=np.uint8)

    for t in range(TARGET_FRAMES):
        if frame_mask[t] == 0:
            continue
        left_points = keypoints[t, LEFT_HAND_SLICE, :].astype(np.float32)
        right_points = keypoints[t, RIGHT_HAND_SLICE, :].astype(np.float32)
        left_scores = scores[t, LEFT_HAND_SLICE].astype(np.float32)
        right_scores = scores[t, RIGHT_HAND_SLICE].astype(np.float32)
        lc, ls, lv, la, _ = compute_hand_center_and_scale(left_points, left_scores)
        rc, rs, rv, ra, _ = compute_hand_center_and_scale(right_points, right_scores)
        left_hand_centers[t] = lc; right_hand_centers[t] = rc
        left_hand_scales[t] = ls; right_hand_scales[t] = rs
        left_hand_areas[t] = la; right_hand_areas[t] = ra
        left_hand_valid[t] = lv; right_hand_valid[t] = rv
        if lv:
            mask = (left_scores >= SCORE_THRESHOLD).astype(np.uint8)
            local = (left_points - lc[None, :]) / (ls + EPS)
            local[mask == 0] = 0.0
            X_hand_local[t, 0:21, :] = local
            hand_local_mask[t, 0:21] = mask
        if rv:
            mask = (right_scores >= SCORE_THRESHOLD).astype(np.uint8)
            local = (right_points - rc[None, :]) / (rs + EPS)
            local[mask == 0] = 0.0
            X_hand_local[t, 21:42, :] = local
            hand_local_mask[t, 21:42] = mask

    return {
        "X_hand_local": X_hand_local,
        "hand_local_mask": hand_local_mask,
        "left_hand_scales": left_hand_scales,
        "right_hand_scales": right_hand_scales,
        "left_hand_areas": left_hand_areas,
        "right_hand_areas": right_hand_areas,
        "left_hand_centers": left_hand_centers,
        "right_hand_centers": right_hand_centers,
        "left_hand_valid": left_hand_valid,
        "right_hand_valid": right_hand_valid,
    }

def compute_motion_score(X_hand_lr_body, hand_mask, hand_slice):
    pts = X_hand_lr_body[:, hand_slice, :]
    mask = hand_mask[:, hand_slice]
    total, steps = 0.0, 0
    for t in range(1, TARGET_FRAMES):
        valid = (mask[t] == 1) & (mask[t - 1] == 1)
        if valid.sum() < 3:
            continue
        step = np.linalg.norm((pts[t] - pts[t - 1])[valid], axis=1).mean()
        total += float(step); steps += 1
    return total / steps if steps else 0.0

def decide_dominant_hand(left_motion, right_motion, left_valid_ratio, right_valid_ratio):
    left_score = 0.65 * left_motion + 0.35 * left_valid_ratio
    right_score = 0.65 * right_motion + 0.35 * right_valid_ratio
    if left_score < EPS and right_score < EPS:
        return "unknown", float(left_score), float(right_score)
    ratio = left_score / (right_score + EPS)
    if ratio > 1.25:
        return "left", float(left_score), float(right_score)
    if ratio < 0.80:
        return "right", float(left_score), float(right_score)
    return "both", float(left_score), float(right_score)

def build_dominant_hand_view(X_hand_local, hand_local_mask, dominant_hand):
    X_dom = np.zeros_like(X_hand_local, dtype=np.float32)
    mask_dom = np.zeros_like(hand_local_mask, dtype=np.uint8)
    left, right = X_hand_local[:, 0:21, :], X_hand_local[:, 21:42, :]
    left_mask, right_mask = hand_local_mask[:, 0:21], hand_local_mask[:, 21:42]
    if dominant_hand == "right":
        X_dom[:, 0:21, :] = right; X_dom[:, 21:42, :] = left
        mask_dom[:, 0:21] = right_mask; mask_dom[:, 21:42] = left_mask
    else:
        X_dom[:, 0:21, :] = left; X_dom[:, 21:42, :] = right
        mask_dom[:, 0:21] = left_mask; mask_dom[:, 21:42] = right_mask
    return (X_dom * mask_dom[..., None]).astype(np.float32), mask_dom.astype(np.uint8)

AUX_V3_NAMES = [
    "frame_valid", "center_valid", "scale_valid", "angle_valid",
    "body_visible_ratio", "face_visible_ratio", "upper49_visible_ratio",
    "left_hand_visible_ratio", "right_hand_visible_ratio", "both_hands_visible",
    "left_hand_local_valid", "right_hand_local_valid",
    "left_hand_scale_ratio", "right_hand_scale_ratio",
    "left_hand_area_ratio", "right_hand_area_ratio",
    "left_hand_box_area_ratio", "right_hand_box_area_ratio",
    "hand_center_distance_body", "left_hand_to_nose_distance", "right_hand_to_nose_distance",
    "left_hand_to_chest_distance", "right_hand_to_chest_distance",
    "body_scale_ratio", "dominant_is_left", "dominant_is_right", "dominant_is_both",
    "shoulders_visible", "elbows_visible", "wrists_visible",
]

MOTION_AUX_NAMES = [
    "motion_valid", "upper_noalign_motion_mean", "upper_noalign_motion_max",
    "upper_aligned_motion_mean", "upper_aligned_motion_max",
    "hand_local_motion_mean", "hand_local_motion_max",
    "hand_dominant_motion_mean", "hand_dominant_motion_max",
    "left_hand_local_motion_mean", "right_hand_local_motion_mean",
    "left_wrist_upper_noalign_speed", "right_wrist_upper_noalign_speed",
    "hand_motion_balance", "valid_motion_ratio",
]

def safe_distance(a, b, valid_a=True, valid_b=True):
    if not valid_a or not valid_b:
        return 0.0, 0
    return float(np.linalg.norm(a - b)), 1

def build_aux_v3(frame_mask, landmark_mask, body_mask, face_mask, upper_mask, X_upper, hand_local_pack, geom, dominant_hand):
    X_aux = np.zeros((TARGET_FRAMES, len(AUX_V3_NAMES)), dtype=np.float32)
    X_aux_mask = np.zeros((TARGET_FRAMES, len(AUX_V3_NAMES)), dtype=np.uint8)
    idx = {name: i for i, name in enumerate(AUX_V3_NAMES)}
    scales = geom["scales"]
    center_valid, scale_valid, angle_valid = geom["center_valid"], geom["scale_valid"], geom["angle_valid"]
    median_scale = float(np.median(scales[scales > EPS])) if np.any(scales > EPS) else 1.0
    lhs = hand_local_pack["left_hand_scales"]
    rhs = hand_local_pack["right_hand_scales"]
    med_lhs = float(np.median(lhs[lhs > EPS])) if np.any(lhs > EPS) else 1.0
    med_rhs = float(np.median(rhs[rhs > EPS])) if np.any(rhs > EPS) else 1.0

    for t in range(TARGET_FRAMES):
        if frame_mask[t] == 0:
            continue
        X_aux[t, idx["frame_valid"]] = 1.0; X_aux_mask[t, idx["frame_valid"]] = 1
        for name, arr in [("center_valid", center_valid), ("scale_valid", scale_valid), ("angle_valid", angle_valid)]:
            X_aux[t, idx[name]] = float(arr[t]); X_aux_mask[t, idx[name]] = 1

        body_ratio = float(body_mask[t].mean())
        face_ratio = float(face_mask[t].mean())
        upper_ratio = float(upper_mask[t].mean())
        left_ratio = float(landmark_mask[t, LEFT_HAND_SLICE].mean())
        right_ratio = float(landmark_mask[t, RIGHT_HAND_SLICE].mean())
        vals = {
            "body_visible_ratio": body_ratio,
            "face_visible_ratio": face_ratio,
            "upper49_visible_ratio": upper_ratio,
            "left_hand_visible_ratio": left_ratio,
            "right_hand_visible_ratio": right_ratio,
            "both_hands_visible": float(left_ratio > 0.30 and right_ratio > 0.30),
            "left_hand_local_valid": float(hand_local_pack["left_hand_valid"][t]),
            "right_hand_local_valid": float(hand_local_pack["right_hand_valid"][t]),
            "body_scale_ratio": float(scales[t] / (median_scale + EPS)),
            "dominant_is_left": float(dominant_hand == "left"),
            "dominant_is_right": float(dominant_hand == "right"),
            "dominant_is_both": float(dominant_hand == "both"),
            "shoulders_visible": float(landmark_mask[t, LEFT_SHOULDER] and landmark_mask[t, RIGHT_SHOULDER]),
            "elbows_visible": float(landmark_mask[t, LEFT_ELBOW] and landmark_mask[t, RIGHT_ELBOW]),
            "wrists_visible": float(landmark_mask[t, LEFT_WRIST_BODY] and landmark_mask[t, RIGHT_WRIST_BODY]),
        }
        for name, val in vals.items():
            X_aux[t, idx[name]] = val; X_aux_mask[t, idx[name]] = 1

        if hand_local_pack["left_hand_valid"][t]:
            X_aux[t, idx["left_hand_scale_ratio"]] = float(lhs[t] / (med_lhs + EPS))
            X_aux[t, idx["left_hand_area_ratio"]] = float(hand_local_pack["left_hand_areas"][t] / ((med_lhs ** 2) + EPS))
            X_aux_mask[t, [idx["left_hand_scale_ratio"], idx["left_hand_area_ratio"]]] = 1
        if hand_local_pack["right_hand_valid"][t]:
            X_aux[t, idx["right_hand_scale_ratio"]] = float(rhs[t] / (med_rhs + EPS))
            X_aux[t, idx["right_hand_area_ratio"]] = float(hand_local_pack["right_hand_areas"][t] / ((med_rhs ** 2) + EPS))
            X_aux_mask[t, [idx["right_hand_scale_ratio"], idx["right_hand_area_ratio"]]] = 1

        nose = X_upper[t, 0]
        chest = (X_upper[t, 1] + X_upper[t, 2]) / 2.0
        left_hand_pts = X_upper[t, LEFT_HAND_UPPER_LOCAL_SLICE, :]
        right_hand_pts = X_upper[t, RIGHT_HAND_UPPER_LOCAL_SLICE, :]
        left_hand_mask = upper_mask[t, LEFT_HAND_UPPER_LOCAL_SLICE]
        right_hand_mask = upper_mask[t, RIGHT_HAND_UPPER_LOCAL_SLICE]
        left_valid = left_hand_mask.sum() >= 3
        right_valid = right_hand_mask.sum() >= 3
        left_center = left_hand_pts[left_hand_mask == 1].mean(axis=0) if left_valid else np.zeros((2,), dtype=np.float32)
        right_center = right_hand_pts[right_hand_mask == 1].mean(axis=0) if right_valid else np.zeros((2,), dtype=np.float32)
        nose_valid = bool(upper_mask[t, 0] == 1)
        shoulder_valid = bool(upper_mask[t, 1] == 1 and upper_mask[t, 2] == 1)
        for name, a, b, va, vb in [
            ("hand_center_distance_body", left_center, right_center, left_valid, right_valid),
            ("left_hand_to_nose_distance", left_center, nose, left_valid, nose_valid),
            ("right_hand_to_nose_distance", right_center, nose, right_valid, nose_valid),
            ("left_hand_to_chest_distance", left_center, chest, left_valid, shoulder_valid),
            ("right_hand_to_chest_distance", right_center, chest, right_valid, shoulder_valid),
        ]:
            d, ok = safe_distance(a, b, va, vb)
            X_aux[t, idx[name]] = d; X_aux_mask[t, idx[name]] = ok

    X_aux = sanitize_numeric(X_aux) * X_aux_mask.astype(np.float32)
    return X_aux.astype(np.float32), X_aux_mask.astype(np.uint8)

def compute_velocity(X, mask, frame_mask, clip_value=10.0):
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask).astype(np.uint8)
    frame_mask = np.asarray(frame_mask).astype(np.uint8)
    vel = np.zeros_like(X, dtype=np.float32)
    vel_mask = np.zeros_like(mask, dtype=np.uint8)
    for t in range(1, X.shape[0]):
        if frame_mask[t] == 0 or frame_mask[t - 1] == 0:
            continue
        valid = (mask[t] == 1) & (mask[t - 1] == 1)
        if valid.sum() == 0:
            continue
        diff = np.clip(X[t] - X[t - 1], -clip_value, clip_value)
        vel[t, valid, :] = diff[valid, :]
        vel_mask[t, valid] = 1
    vel = sanitize_numeric(vel) * vel_mask[..., None]
    return vel.astype(np.float32), vel_mask.astype(np.uint8)

def speed_from_velocity(vel):
    return np.linalg.norm(vel.astype(np.float32), axis=-1).astype(np.float32)

def mean_speed_per_frame(vel, vel_mask, node_slice):
    speed = speed_from_velocity(vel[:, node_slice, :])
    mask = vel_mask[:, node_slice]
    out = np.zeros((vel.shape[0],), dtype=np.float32)
    out_mask = np.zeros((vel.shape[0],), dtype=np.uint8)
    for t in range(vel.shape[0]):
        if mask[t].sum() > 0:
            out[t] = float(speed[t][mask[t] == 1].mean())
            out_mask[t] = 1
    return out, out_mask

def max_speed_per_frame(vel, vel_mask, node_slice):
    speed = speed_from_velocity(vel[:, node_slice, :])
    mask = vel_mask[:, node_slice]
    out = np.zeros((vel.shape[0],), dtype=np.float32)
    out_mask = np.zeros((vel.shape[0],), dtype=np.uint8)
    for t in range(vel.shape[0]):
        if mask[t].sum() > 0:
            out[t] = float(speed[t][mask[t] == 1].max())
            out_mask[t] = 1
    return out, out_mask

def point_speed_per_frame(vel, vel_mask, point_idx):
    speed = speed_from_velocity(vel[:, point_idx:point_idx + 1, :])[:, 0]
    mask = vel_mask[:, point_idx]
    out = np.zeros((vel.shape[0],), dtype=np.float32)
    out_mask = np.zeros((vel.shape[0],), dtype=np.uint8)
    valid = mask == 1
    out[valid] = speed[valid]
    out_mask[valid] = 1
    return out, out_mask

def build_motion_aux(frame_mask, upper_noalign_vel, upper_noalign_vel_mask, upper_aligned_vel, upper_aligned_vel_mask, hand_local_vel, hand_local_vel_mask, hand_dominant_vel, hand_dominant_vel_mask):
    X_motion_aux = np.zeros((TARGET_FRAMES, len(MOTION_AUX_NAMES)), dtype=np.float32)
    X_motion_aux_mask = np.zeros((TARGET_FRAMES, len(MOTION_AUX_NAMES)), dtype=np.uint8)
    idx = {name: i for i, name in enumerate(MOTION_AUX_NAMES)}

    uno_mean, uno_mean_m = mean_speed_per_frame(upper_noalign_vel, upper_noalign_vel_mask, slice(0, 49))
    uno_max, uno_max_m = max_speed_per_frame(upper_noalign_vel, upper_noalign_vel_mask, slice(0, 49))
    ua_mean, ua_mean_m = mean_speed_per_frame(upper_aligned_vel, upper_aligned_vel_mask, slice(0, 49))
    ua_max, ua_max_m = max_speed_per_frame(upper_aligned_vel, upper_aligned_vel_mask, slice(0, 49))
    hl_mean, hl_mean_m = mean_speed_per_frame(hand_local_vel, hand_local_vel_mask, slice(0, 42))
    hl_max, hl_max_m = max_speed_per_frame(hand_local_vel, hand_local_vel_mask, slice(0, 42))
    hd_mean, hd_mean_m = mean_speed_per_frame(hand_dominant_vel, hand_dominant_vel_mask, slice(0, 42))
    hd_max, hd_max_m = max_speed_per_frame(hand_dominant_vel, hand_dominant_vel_mask, slice(0, 42))
    left_mean, left_mean_m = mean_speed_per_frame(hand_local_vel, hand_local_vel_mask, slice(0, 21))
    right_mean, right_mean_m = mean_speed_per_frame(hand_local_vel, hand_local_vel_mask, slice(21, 42))
    left_wrist_speed, left_wrist_m = point_speed_per_frame(upper_noalign_vel, upper_noalign_vel_mask, 5)
    right_wrist_speed, right_wrist_m = point_speed_per_frame(upper_noalign_vel, upper_noalign_vel_mask, 6)

    assignments = [
        ("upper_noalign_motion_mean", uno_mean, uno_mean_m), ("upper_noalign_motion_max", uno_max, uno_max_m),
        ("upper_aligned_motion_mean", ua_mean, ua_mean_m), ("upper_aligned_motion_max", ua_max, ua_max_m),
        ("hand_local_motion_mean", hl_mean, hl_mean_m), ("hand_local_motion_max", hl_max, hl_max_m),
        ("hand_dominant_motion_mean", hd_mean, hd_mean_m), ("hand_dominant_motion_max", hd_max, hd_max_m),
        ("left_hand_local_motion_mean", left_mean, left_mean_m), ("right_hand_local_motion_mean", right_mean, right_mean_m),
        ("left_wrist_upper_noalign_speed", left_wrist_speed, left_wrist_m), ("right_wrist_upper_noalign_speed", right_wrist_speed, right_wrist_m),
    ]

    for t in range(TARGET_FRAMES):
        if frame_mask[t] == 0:
            continue
        X_motion_aux[t, idx["motion_valid"]] = 1.0
        X_motion_aux_mask[t, idx["motion_valid"]] = 1
        for name, arr, arr_mask in assignments:
            if arr_mask[t] == 1:
                X_motion_aux[t, idx[name]] = float(arr[t])
                X_motion_aux_mask[t, idx[name]] = 1
        if left_mean_m[t] == 1 and right_mean_m[t] == 1:
            denom = left_mean[t] + right_mean[t] + EPS
            X_motion_aux[t, idx["hand_motion_balance"]] = float((left_mean[t] - right_mean[t]) / denom)
            X_motion_aux_mask[t, idx["hand_motion_balance"]] = 1
        valid_motion_ratio = (upper_noalign_vel_mask[t].mean() + hand_local_vel_mask[t].mean() + hand_dominant_vel_mask[t].mean()) / 3.0
        X_motion_aux[t, idx["valid_motion_ratio"]] = float(valid_motion_ratio)
        X_motion_aux_mask[t, idx["valid_motion_ratio"]] = 1

    X_motion_aux = sanitize_numeric(X_motion_aux) * X_motion_aux_mask.astype(np.float32)
    return X_motion_aux.astype(np.float32), X_motion_aux_mask.astype(np.uint8)


def standardize_aux(X_aux, X_aux_mask):
    aux_mean, aux_std = load_aux_stats()
    X = (X_aux.astype(np.float32) - aux_mean[None, :]) / aux_std[None, :]
    X = sanitize_numeric(X, clip_value=20.0)
    X = X * X_aux_mask.astype(np.float32)
    return X.astype(np.float32)

def make_10A_features_from_64_raw(keypoints64, scores64, frame_mask64):
    keypoints64 = sanitize_numeric(keypoints64)
    scores64 = sanitize_numeric(scores64)
    frame_mask64 = frame_mask64.astype(np.uint8)

    landmark_mask, body_mask, face_mask, _, _ = extract_or_create_masks(keypoints64, scores64, frame_mask64)
    geom = compute_robust_center_scale_angle(keypoints64, scores64, landmark_mask, frame_mask64)

    X_body_aligned_all = body_normalize_keypoints(keypoints64, geom["centers"], geom["scales"], geom["angles"], landmark_mask, frame_mask64, align_shoulders=True)
    X_body_noalign_all = body_normalize_keypoints(keypoints64, geom["centers"], geom["scales"], geom["angles"], landmark_mask, frame_mask64, align_shoulders=False)
    X_upper_aligned, upper_mask = extract_upper49(X_body_aligned_all, landmark_mask)
    X_upper_noalign, _ = extract_upper49(X_body_noalign_all, landmark_mask)

    X_hand_lr_body, hand_lr_body_mask = extract_lr_hands_from_body_norm(X_body_aligned_all, landmark_mask)
    hand_local_pack = hand_local_normalize_raw(keypoints64, scores64, frame_mask64)
    X_hand_local = hand_local_pack["X_hand_local"]
    hand_local_mask = hand_local_pack["hand_local_mask"]

    left_motion = compute_motion_score(X_hand_lr_body, hand_lr_body_mask, slice(0, 21))
    right_motion = compute_motion_score(X_hand_lr_body, hand_lr_body_mask, slice(21, 42))
    real_frames = frame_mask64 > 0
    if real_frames.sum() > 0:
        left_valid_ratio = float((hand_local_mask[real_frames, 0:21].mean(axis=1) > 0.30).mean())
        right_valid_ratio = float((hand_local_mask[real_frames, 21:42].mean(axis=1) > 0.30).mean())
    else:
        left_valid_ratio = right_valid_ratio = 0.0
    dominant_hand, dominant_left_score, dominant_right_score = decide_dominant_hand(left_motion, right_motion, left_valid_ratio, right_valid_ratio)
    X_hand_dominant, hand_dominant_mask = build_dominant_hand_view(X_hand_local, hand_local_mask, dominant_hand)

    X_aux_v3, X_aux_v3_mask = build_aux_v3(frame_mask64, landmark_mask, body_mask, face_mask, upper_mask, X_upper_aligned, hand_local_pack, geom, dominant_hand)

    X_upper_noalign_velocity, upper_noalign_velocity_mask = compute_velocity(X_upper_noalign, upper_mask, frame_mask64)
    X_upper_aligned_velocity, upper_aligned_velocity_mask = compute_velocity(X_upper_aligned, upper_mask, frame_mask64)
    X_hand_local_velocity, hand_local_velocity_mask = compute_velocity(X_hand_local, hand_local_mask, frame_mask64)
    X_hand_dominant_velocity, hand_dominant_velocity_mask = compute_velocity(X_hand_dominant, hand_dominant_mask, frame_mask64)

    X_motion_aux, X_motion_aux_mask = build_motion_aux(
        frame_mask64,
        X_upper_noalign_velocity,
        upper_noalign_velocity_mask,
        X_upper_aligned_velocity,
        upper_aligned_velocity_mask,
        X_hand_local_velocity,
        hand_local_velocity_mask,
        X_hand_dominant_velocity,
        hand_dominant_velocity_mask,
    )
    X_aux_10A = np.concatenate([X_aux_v3, X_motion_aux], axis=1).astype(np.float32)
    X_aux_10A_mask = np.concatenate([X_aux_v3_mask, X_motion_aux_mask], axis=1).astype(np.uint8)
    if X_aux_10A.shape[1] != AUX_DIM:
        raise ValueError(f"AUX_DIM mismatch: built {X_aux_10A.shape[1]} expected {AUX_DIM}")
    X_aux_10A = standardize_aux(X_aux_10A, X_aux_10A_mask)

    return {
        "X_hand_local": X_hand_local,
        "X_hand_local_velocity": X_hand_local_velocity,
        "hand_local_mask": hand_local_mask,
        "hand_local_velocity_mask": hand_local_velocity_mask,
        "X_hand_dominant": X_hand_dominant,
        "X_hand_dominant_velocity": X_hand_dominant_velocity,
        "hand_dominant_mask": hand_dominant_mask,
        "hand_dominant_velocity_mask": hand_dominant_velocity_mask,
        "X_upper_noalign": X_upper_noalign,
        "X_upper_noalign_velocity": X_upper_noalign_velocity,
        "upper_noalign_mask": upper_mask,
        "upper_noalign_velocity_mask": upper_noalign_velocity_mask,
        "X_upper_aligned": X_upper_aligned,
        "X_upper_aligned_velocity": X_upper_aligned_velocity,
        "upper_aligned_mask": upper_mask,
        "upper_aligned_velocity_mask": upper_aligned_velocity_mask,
        "X_aux_10A": X_aux_10A,
        "X_aux_10A_mask": X_aux_10A_mask,
        "frame_mask": frame_mask64.astype(np.uint8),
        "dominant_hand": dominant_hand,
        "quality": {
            "real_frames": int(frame_mask64.sum()),
            "upper_valid_ratio": float(upper_mask[real_frames].mean()) if real_frames.sum() else 0.0,
            "hand_valid_ratio": float(hand_local_mask[real_frames].mean()) if real_frames.sum() else 0.0,
            "left_valid_ratio": float(left_valid_ratio),
            "right_valid_ratio": float(right_valid_ratio),
            "body_valid_ratio": float(body_mask[real_frames].mean()) if real_frames.sum() else 0.0,
            "dominant_left_score": float(dominant_left_score),
            "dominant_right_score": float(dominant_right_score),
        },
    }

def mirror_raw_wholebody(keypoints, scores=None):
    kp = np.asarray(keypoints, dtype=np.float32).copy()
    kp[..., 0] *= -1.0
    kp = kp[:, WHOLEBODY_SWAP, :]
    if scores is None:
        return kp, None
    sc = np.asarray(scores, dtype=np.float32).copy()
    sc = sc[:, WHOLEBODY_SWAP]
    return kp, sc

def build_windows(keypoints, scores, frame_mask=None, window=TARGET_FRAMES, stride=WINDOW_STRIDE):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    T = keypoints.shape[0]
    if frame_mask is None:
        frame_mask = np.ones((T,), dtype=np.uint8)
    else:
        frame_mask = np.asarray(frame_mask, dtype=np.uint8)
    windows = []
    if T <= window:
        kp64, sc64, fm64, idx = ensure_64_frames(keypoints, scores, frame_mask)
        windows.append({"start": 0, "end": T, "indices": idx, "keypoints": kp64, "scores": sc64, "frame_mask": fm64})
        return windows

    starts = list(range(0, max(T - window + 1, 1), stride))
    if starts[-1] != T - window:
        starts.append(T - window)
    for st in starts:
        en = st + window
        windows.append({
            "start": int(st),
            "end": int(en),
            "indices": np.arange(st, en),
            "keypoints": keypoints[st:en],
            "scores": scores[st:en],
            "frame_mask": frame_mask[st:en],
        })
    return windows

def features_to_torch_batch(features, device):
    import torch

    tensor_keys_float = [
        "X_hand_local", "X_hand_local_velocity", "X_hand_dominant", "X_hand_dominant_velocity",
        "X_upper_noalign", "X_upper_noalign_velocity", "X_upper_aligned", "X_upper_aligned_velocity",
        "X_aux_10A",
    ]
    tensor_keys_mask = [
        "hand_local_mask", "hand_local_velocity_mask", "hand_dominant_mask", "hand_dominant_velocity_mask",
        "upper_noalign_mask", "upper_noalign_velocity_mask", "upper_aligned_mask", "upper_aligned_velocity_mask",
        "X_aux_10A_mask", "frame_mask",
    ]
    batch = {}
    for k in tensor_keys_float:
        batch[k] = torch.from_numpy(features[k]).unsqueeze(0).float().to(device)
    for k in tensor_keys_mask:
        batch[k] = torch.from_numpy(features[k].astype(np.float32)).unsqueeze(0).float().to(device)
    return batch

# ---------------- RTMPose extraction ----------------

def build_rtmpose_inferencer(device=RTMPOSE_DEVICE):
    if not MMPOSE_AVAILABLE:
        raise ImportError(
            "mmpose is not available. Install/run inside your RTMPose environment. "
            f"Original error: {MMPOSE_IMPORT_ERROR}"
        )
    return MMPoseInferencer(pose2d="wholebody", device=device)


def get_rtmpose_inferencer():
    global _RTMPOSE_INFERENCER
    if _RTMPOSE_INFERENCER is None:
        _RTMPOSE_INFERENCER = build_rtmpose_inferencer()
    return _RTMPOSE_INFERENCER


def safe_open_video(video_path):
    video_path = str(video_path)
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV cv2 is not available.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def get_video_info(video_path):
    cap = safe_open_video(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = total_frames / fps if fps > 0 else None
    return {
        "video_path": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def get_instance_score(instance):
    if instance is None:
        return 0.0
    if "bbox_score" in instance:
        try:
            return float(np.mean(instance["bbox_score"]))
        except Exception:
            pass
    if "keypoint_scores" in instance:
        scores = np.asarray(instance["keypoint_scores"], dtype=np.float32)
        if scores.size > 0:
            return float(np.nanmean(scores))
    return 0.0


def choose_best_person_instance(predictions):
    if predictions is None or len(predictions) == 0:
        return None
    instances = predictions[0]
    if instances is None or len(instances) == 0:
        return None
    return max(instances, key=get_instance_score)


def extract_wholebody_from_frame(frame_rgb, inferencer, score_threshold=SCORE_THRESHOLD):
    keypoints = np.zeros((NUM_WHOLEBODY_KEYPOINTS, 2), dtype=np.float32)
    scores = np.zeros((NUM_WHOLEBODY_KEYPOINTS,), dtype=np.float32)
    mask = np.zeros((NUM_WHOLEBODY_KEYPOINTS,), dtype=np.float32)

    if frame_rgb is None:
        return keypoints, scores, mask, False

    result_generator = inferencer(frame_rgb, show=False, return_vis=False, kpt_thr=score_threshold)
    result = next(result_generator)
    best_instance = choose_best_person_instance(result.get("predictions", []))
    if best_instance is None:
        return keypoints, scores, mask, False

    pred_keypoints = np.asarray(best_instance.get("keypoints", []), dtype=np.float32)
    pred_scores = np.asarray(best_instance.get("keypoint_scores", []), dtype=np.float32)

    if pred_keypoints.ndim != 2 or pred_keypoints.shape[0] == 0:
        return keypoints, scores, mask, False

    n = min(NUM_WHOLEBODY_KEYPOINTS, pred_keypoints.shape[0])
    keypoints[:n] = pred_keypoints[:n, :2]
    if pred_scores.size >= n:
        scores[:n] = pred_scores[:n]
    else:
        scores[:n] = 1.0
    mask[:n] = (scores[:n] >= score_threshold).astype(np.float32)
    detected = bool(mask.sum() > 0)
    return keypoints, scores, mask, detected


def extract_rtmpose_keypoints_from_video(video_path, output_path=None, frame_step=EXTRACT_FRAME_STEP, max_extract_frames=MAX_EXTRACT_FRAMES):
    video_path = Path(video_path)
    info = get_video_info(video_path)
    if output_path is None:
        output_path = KEYPOINTS_DIR / f"{video_path.stem}_rtmpose_wholebody.npz"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inferencer = get_rtmpose_inferencer()
    cap = safe_open_video(video_path)
    total_frames = info["total_frames"]

    frame_indices = list(range(0, total_frames, max(int(frame_step), 1)))
    if max_extract_frames is not None:
        frame_indices = frame_indices[: int(max_extract_frames)]

    keypoints_list, scores_list, frame_mask_list, landmark_mask_list, sampled_indices = [], [], [], [], []
    detected_frames = body_detected_frames = left_hand_detected_frames = right_hand_detected_frames = 0
    failed_read_frames = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame_bgr = cap.read()
        sampled_indices.append(int(idx))

        if not ret or frame_bgr is None:
            failed_read_frames += 1
            keypoints = np.zeros((NUM_WHOLEBODY_KEYPOINTS, 2), dtype=np.float32)
            scores = np.zeros((NUM_WHOLEBODY_KEYPOINTS,), dtype=np.float32)
            landmark_mask = np.zeros((NUM_WHOLEBODY_KEYPOINTS,), dtype=np.float32)
            frame_mask = 0
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            try:
                keypoints, scores, landmark_mask, detected = extract_wholebody_from_frame(frame_rgb, inferencer)
            except Exception:
                keypoints = np.zeros((NUM_WHOLEBODY_KEYPOINTS, 2), dtype=np.float32)
                scores = np.zeros((NUM_WHOLEBODY_KEYPOINTS,), dtype=np.float32)
                landmark_mask = np.zeros((NUM_WHOLEBODY_KEYPOINTS,), dtype=np.float32)
                detected = False
            frame_mask = 1
            if detected:
                detected_frames += 1
            if landmark_mask[BODY_SLICE].sum() > 0:
                body_detected_frames += 1
            if landmark_mask[LEFT_HAND_SLICE].sum() > 0:
                left_hand_detected_frames += 1
            if landmark_mask[RIGHT_HAND_SLICE].sum() > 0:
                right_hand_detected_frames += 1

        keypoints_list.append(keypoints)
        scores_list.append(scores)
        landmark_mask_list.append(landmark_mask)
        frame_mask_list.append(frame_mask)

    cap.release()

    if len(keypoints_list) == 0:
        raise RuntimeError(f"No frames extracted from video: {video_path}")

    keypoints = np.stack(keypoints_list).astype(np.float32)
    keypoint_scores = np.stack(scores_list).astype(np.float32)
    frame_mask = np.asarray(frame_mask_list, dtype=np.uint8)
    landmark_mask = np.stack(landmark_mask_list).astype(np.float32)
    sampled_indices = np.asarray(sampled_indices, dtype=np.int32)

    real_frames = int(frame_mask.sum())
    denom = max(real_frames, 1)
    report = {
        "video_path": str(video_path),
        "keypoints_path": str(output_path),
        "total_video_frames": int(total_frames),
        "extracted_frames": int(len(frame_mask)),
        "real_frames": real_frames,
        "failed_read_frames": int(failed_read_frames),
        "fps": float(info["fps"]),
        "width": int(info["width"]),
        "height": int(info["height"]),
        "frame_step": int(frame_step),
        "score_threshold": float(SCORE_THRESHOLD),
        "detection_ratio": float(detected_frames / denom),
        "body_detection_ratio": float(body_detected_frames / denom),
        "left_hand_detection_ratio": float(left_hand_detected_frames / denom),
        "right_hand_detection_ratio": float(right_hand_detected_frames / denom),
    }

    np.savez_compressed(
        output_path,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        frame_mask=frame_mask,
        landmark_mask=landmark_mask,
        sampled_indices=sampled_indices,
        extraction_report=json.dumps(report, ensure_ascii=False),
    )
    return output_path, report


def resolve_input_to_npz(input_path: Path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    suffix = input_path.suffix.lower()

    if suffix == ".npz":
        return input_path, {"input_type": "npz", "extraction_skipped": True}

    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported input type: {suffix}")

    output_path = KEYPOINTS_DIR / f"{input_path.stem}_rtmpose_wholebody.npz"
    npz_path, report = extract_rtmpose_keypoints_from_video(input_path, output_path=output_path)
    report["input_type"] = "video"
    report["extraction_skipped"] = False
    return npz_path, report


def preprocess_video(input_path: Path, stride=WINDOW_STRIDE, use_mirror_tta=USE_MIRROR_TTA) -> Dict:
    """FastAPI-friendly preprocessing function.

    Input: video path or existing .npz path.
    Output: candidates list. Each candidate has features ready for the model.
    """
    npz_path, extraction_report = resolve_input_to_npz(Path(input_path))
    keypoints, scores, frame_mask, _ = load_raw_keypoints_npz(npz_path)
    windows = build_windows(keypoints, scores, frame_mask, window=TARGET_FRAMES, stride=stride)

    candidates: List[Dict] = []
    for window_id, w in enumerate(windows):
        features = make_10A_features_from_64_raw(w["keypoints"], w["scores"], w["frame_mask"])
        candidates.append({
            "window_id": window_id,
            "start": int(w["start"]),
            "end": int(w["end"]),
            "path": "original",
            "features": features,
        })

        if use_mirror_tta:
            mk, ms = mirror_raw_wholebody(w["keypoints"], w["scores"])
            mfeatures = make_10A_features_from_64_raw(mk, ms, w["frame_mask"])
            candidates.append({
                "window_id": window_id,
                "start": int(w["start"]),
                "end": int(w["end"]),
                "path": "mirror",
                "features": mfeatures,
            })

    return {
        "input_path": str(input_path),
        "npz_path": str(npz_path),
        "num_windows": len(windows),
        "candidates": candidates,
        "extraction_report": extraction_report,
    }