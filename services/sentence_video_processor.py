from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4
import shutil

import cv2

try:
    from config import TEMP_DIR
    from services.video_processor import preprocess_video
    from services.sign_predictor import predict_sign
except ImportError:  # Allows running as App.main:app from project root
    from config import TEMP_DIR
    from services.video_processor import preprocess_video
    from services.sign_predictor import predict_sign


DEFAULT_WINDOW_SECONDS = 2.2   # ~64 frames at 30 FPS
DEFAULT_STRIDE_SECONDS = 1.1   # 50% overlap
DEFAULT_MIN_SEGMENT_SECONDS = 0.7
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_MIN_MARGIN = 0.08
DEFAULT_MAX_SEGMENTS = 40


class SentenceVideoError(Exception):
    """Raised when sentence-video processing fails."""


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def get_video_info(video_path: Path) -> Dict[str, float]:
    """Return basic video metadata using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SentenceVideoError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        fps = 30.0

    duration = frame_count / fps if frame_count > 0 else 0.0

    return {
        "fps": float(fps),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "duration": float(duration),
    }


def write_video_segment(
    input_video_path: Path,
    output_video_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    width: int,
    height: int,
) -> int:
    """
    Write [start_frame, end_frame) segment to output_video_path.
    Returns number of frames actually written.
    """
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise SentenceVideoError(f"Could not open video: {input_video_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4v is generally readable by OpenCV and enough for internal temp clips.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise SentenceVideoError(f"Could not create segment video: {output_video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    written = 0
    current = start_frame

    while current < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if frame is None:
            break

        writer.write(frame)
        written += 1
        current += 1

    writer.release()
    cap.release()

    return written


def create_overlapping_segments(
    video_path: Path,
    output_dir: Optional[Path] = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> Tuple[List[Dict], Path, Dict[str, float]]:
    """
    Split a full sentence video into overlapping clips.

    Returns:
        segments: list of dicts containing segment path and frame/time range
        work_dir: temp folder used for the generated segments
        video_info: original video metadata
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)

    fps = float(info["fps"])
    frame_count = int(info["frame_count"])
    width = int(info["width"])
    height = int(info["height"])

    if frame_count <= 0:
        raise SentenceVideoError("Video has no readable frames.")

    if width <= 0 or height <= 0:
        raise SentenceVideoError("Video width/height could not be detected.")

    window_frames = max(1, int(round(window_seconds * fps)))
    stride_frames = max(1, int(round(stride_seconds * fps)))
    min_segment_frames = max(1, int(round(min_segment_seconds * fps)))

    if output_dir is None:
        output_dir = Path(TEMP_DIR) / "sentence_segments" / uuid4().hex

    output_dir.mkdir(parents=True, exist_ok=True)

    segments: List[Dict] = []
    start = 0
    index = 0

    while start < frame_count and len(segments) < max_segments:
        end = min(start + window_frames, frame_count)

        if (end - start) < min_segment_frames:
            break

        segment_path = output_dir / f"segment_{index:04d}.mp4"

        written = write_video_segment(
            input_video_path=video_path,
            output_video_path=segment_path,
            start_frame=start,
            end_frame=end,
            fps=fps,
            width=width,
            height=height,
        )

        if written >= min_segment_frames:
            segments.append({
                "segment_index": index,
                "segment_path": str(segment_path),
                "start_frame": int(start),
                "end_frame": int(end),
                "start_sec": float(start / fps),
                "end_sec": float(end / fps),
                "written_frames": int(written),
            })
            index += 1

        if end >= frame_count:
            break

        start += stride_frames

    if not segments:
        raise SentenceVideoError("No valid segments were created from the video.")

    return segments, output_dir, info


def _extract_prediction_values(prediction: Dict) -> Tuple[str, float, float]:
    """Normalize output keys from predict_sign()."""
    word = str(prediction.get("word") or prediction.get("arabic_label") or "").strip()
    confidence = _safe_float(
        prediction.get("confidence", prediction.get("top1_prob", prediction.get("probability", 0.0))),
        0.0,
    )
    margin = _safe_float(prediction.get("margin", 1.0), 1.0)
    return word, confidence, margin


def clean_segment_predictions(
    segment_predictions: List[Dict],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> List[str]:
    """
    Convert noisy overlapping-window predictions into a clean word list.

    Example:
        [أنا, أنا, رايح, رايح, مدرسة]
    becomes:
        [أنا, رايح, مدرسة]
    """
    words: List[str] = []
    last_word: Optional[str] = None

    for item in segment_predictions:
        word = str(item.get("word", "")).strip()
        confidence = _safe_float(item.get("confidence", 0.0), 0.0)
        margin = _safe_float(item.get("margin", 1.0), 1.0)

        if not word:
            continue

        if confidence < min_confidence:
            continue

        if margin < min_margin:
            continue

        if word == last_word:
            continue

        words.append(word)
        last_word = word

    return words


def predict_sentence_from_video(
    video_path: Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    keep_debug_segments: bool = False,
) -> Dict:
    """
    Full pipeline:
        full video -> segments -> preprocess each segment -> predict each segment -> clean words
    """
    work_dir: Optional[Path] = None

    try:
        segments, work_dir, video_info = create_overlapping_segments(
            video_path=Path(video_path),
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            max_segments=max_segments,
        )

        segment_predictions: List[Dict] = []

        for segment in segments:
            segment_path = Path(segment["segment_path"])

            preprocessed = preprocess_video(segment_path)
            prediction = predict_sign(preprocessed)

            word, confidence, margin = _extract_prediction_values(prediction)

            segment_predictions.append({
                "segment_index": segment["segment_index"],
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "word": word,
                "confidence": confidence,
                "margin": margin,
                "prediction": prediction,
            })

        raw_words = clean_segment_predictions(
            segment_predictions,
            min_confidence=min_confidence,
            min_margin=min_margin,
        )

        return {
            "raw_words": raw_words,
            "raw_sentence": " - ".join(raw_words),
            "video_info": video_info,
            "window_seconds": float(window_seconds),
            "stride_seconds": float(stride_seconds),
            "min_confidence": float(min_confidence),
            "min_margin": float(min_margin),
            "segments": segment_predictions,
            "debug_segments_dir": str(work_dir) if keep_debug_segments and work_dir else None,
        }

    finally:
        if work_dir is not None and not keep_debug_segments:
            shutil.rmtree(work_dir, ignore_errors=True)
