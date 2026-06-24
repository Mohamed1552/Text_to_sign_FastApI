from pathlib import Path
from typing import Dict, List, Any, Optional
from uuid import uuid4
import cv2

from services.video_processor import preprocess_video
from services.sign_predictor import predict_sign


# Faster / more practical defaults for full sentence videos.
# Old values like window=2.2, stride=1.1, max_segments=40 were too slow.
WINDOW_SECONDS_DEFAULT = 3.0
STRIDE_SECONDS_DEFAULT = 1.5
MAX_SEGMENTS_DEFAULT = 12
MIN_CONFIDENCE_DEFAULT = 0.20
MIN_MARGIN_DEFAULT = 0.03
MIN_TIME_GAP_DEFAULT = 1.2


def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Read basic video metadata."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if not fps or fps <= 0:
        fps = 30.0

    duration = frame_count / fps if fps > 0 else 0.0

    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": float(duration),
    }


def cut_video_segment(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
) -> Path:
    """Cut a short video segment using OpenCV."""
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame + 1, int(end_sec * fps))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame
    while current_frame < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()

    return output_path


def extract_prediction_summary(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the real word/confidence/margin from predict_sign().

    Important:
    Your predict_sign() response stores the useful confidence inside:
        prediction["best"]["top1_prob"]
        prediction["best"]["margin"]

    Not necessarily directly in:
        prediction["confidence"]
    """
    if not isinstance(prediction, dict):
        return {
            "word": "",
            "confidence": 0.0,
            "margin": 0.0,
            "selection_score": 0.0,
            "topk": [],
        }

    best = prediction.get("best", {}) or {}
    if not isinstance(best, dict):
        best = {}

    word = (
        best.get("word")
        or best.get("arabic_label")
        or prediction.get("word")
        or ""
    )

    confidence = (
        best.get("top1_prob")
        or best.get("confidence")
        or prediction.get("confidence")
        or prediction.get("top1_prob")
        or prediction.get("probability")
        or 0.0
    )

    margin = (
        best.get("margin")
        or prediction.get("margin")
        or 0.0
    )

    selection_score = (
        best.get("selection_score")
        or prediction.get("selection_score")
        or 0.0
    )

    topk = best.get("topk") or prediction.get("topk") or []

    return {
        "word": str(word).strip(),
        "confidence": float(confidence),
        "margin": float(margin),
        "selection_score": float(selection_score),
        "topk": topk,
    }


def build_words_from_segments(
    segment_predictions,
    min_confidence=0.30,
    min_margin=0.08,
    min_time_gap=3.0,
):
    """
    Reduce repeated words from overlapping segments.

    Idea:
    - Ignore weak predictions.
    - If the same word appears again within min_time_gap seconds, skip it.
    - If the same word appears with better confidence inside the same time group,
      keep only the stronger one.
    """

    final_items = []

    for item in segment_predictions:
        word = str(item.get("word", "")).strip()
        confidence = float(item.get("confidence", 0.0))
        margin = float(item.get("margin", 0.0))
        start_sec = float(item.get("start_sec", 0.0))

        if not word:
            continue

        if confidence < min_confidence:
            continue

        if margin < min_margin:
            continue

        if not final_items:
            final_items.append({
                "word": word,
                "confidence": confidence,
                "margin": margin,
                "start_sec": start_sec,
            })
            continue

        last = final_items[-1]
        same_word = word == last["word"]
        close_time = (start_sec - last["start_sec"]) < min_time_gap

        # Same word repeated shortly after → treat as duplicate
        if same_word and close_time:
            # Keep the stronger one only
            if confidence > last["confidence"]:
                final_items[-1] = {
                    "word": word,
                    "confidence": confidence,
                    "margin": margin,
                    "start_sec": start_sec,
                }
            continue

        final_items.append({
            "word": word,
            "confidence": confidence,
            "margin": margin,
            "start_sec": start_sec,
        })

    return [item["word"] for item in final_items]


def make_compact_segments(segment_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return small response objects instead of huge nested prediction JSON."""
    compact = []

    for item in segment_predictions:
        compact.append({
            "segment_index": item.get("segment_index"),
            "start_sec": item.get("start_sec"),
            "end_sec": item.get("end_sec"),
            "word": item.get("word"),
            "confidence": item.get("confidence", 0.0),
            "margin": item.get("margin", 0.0),
            "selection_score": item.get("selection_score", 0.0),
            "error": item.get("error"),
        })

    return compact


def predict_full_sentence_video(
    video_path: Path,
    temp_dir: Path,
    window_seconds: float = WINDOW_SECONDS_DEFAULT,
    stride_seconds: float = STRIDE_SECONDS_DEFAULT,
    min_confidence: float = MIN_CONFIDENCE_DEFAULT,
    min_margin: float = MIN_MARGIN_DEFAULT,
    max_segments: int = MAX_SEGMENTS_DEFAULT,
    min_time_gap: float = MIN_TIME_GAP_DEFAULT,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Approximate full-sentence video pipeline.

    Input:
        One video containing multiple signs.

    Current method:
        full video -> time windows -> predict every window -> clean nearby duplicates.

    Important limitation:
        The model is word-level, so this is approximate. For best accuracy,
        use frontend chunks or later replace this with motion-based segmentation.
    """
    video_path = Path(video_path)
    temp_dir = Path(temp_dir)

    info = get_video_info(video_path)
    duration = float(info["duration"])
    fps = float(info["fps"])

    if duration <= 0:
        raise ValueError("Invalid video duration")

    if window_seconds <= 0:
        window_seconds = WINDOW_SECONDS_DEFAULT

    if stride_seconds <= 0:
        stride_seconds = window_seconds

    if max_segments <= 0:
        max_segments = MAX_SEGMENTS_DEFAULT

    request_id = uuid4().hex
    segments_dir = temp_dir / "sentence_segments" / request_id
    segments_dir.mkdir(parents=True, exist_ok=True)

    segment_predictions: List[Dict[str, Any]] = []

    start = 0.0
    segment_index = 0

    while start < duration and segment_index < max_segments:
        end = min(start + window_seconds, duration)

        # Ignore tiny trailing segments.
        if end - start < 0.60:
            break

        segment_path = segments_dir / f"segment_{segment_index:04d}.mp4"

        start_frame = int(start * fps)
        end_frame = int(end * fps)

        cut_video_segment(
            video_path=video_path,
            output_path=segment_path,
            start_sec=start,
            end_sec=end,
        )

        try:
            preprocessed = preprocess_video(segment_path)
            prediction = predict_sign(preprocessed)
            summary = extract_prediction_summary(prediction)

            item = {
                "segment_index": segment_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "segment_file": str(segment_path),
                "word": summary["word"],
                "confidence": summary["confidence"],
                "margin": summary["margin"],
                "selection_score": summary["selection_score"],
                "topk": summary["topk"],
            }

            if debug:
                item["prediction"] = prediction

            segment_predictions.append(item)

        except Exception as e:
            segment_predictions.append({
                "segment_index": segment_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "segment_file": str(segment_path),
                "word": "",
                "confidence": 0.0,
                "margin": 0.0,
                "selection_score": 0.0,
                "topk": [],
                "error": repr(e),
            })

        segment_index += 1
        start += stride_seconds

    raw_words = build_words_from_segments(
        segment_predictions=segment_predictions,
        min_confidence=min_confidence,
        min_margin=min_margin,
        min_time_gap=min_time_gap,
    )

    compact_segments = make_compact_segments(segment_predictions)

    return {
        "video_info": info,
        "settings": {
            "window_seconds": window_seconds,
            "stride_seconds": stride_seconds,
            "min_confidence": min_confidence,
            "min_margin": min_margin,
            "min_time_gap": min_time_gap,
            "max_segments": max_segments,
            "debug": debug,
        },
        "raw_words": raw_words,
        "compact_segments": compact_segments,
        "segments": segment_predictions if debug else compact_segments,
    }
