from pathlib import Path
from typing import Optional
from uuid import uuid4
import shutil

from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException

try:
    from config import UPLOAD_DIR, VIDEO_EXTENSIONS
    from services.full_sentence_video_service import predict_sentence_from_video
    from services.nlp_refiner import refine_to_egyptian
except ImportError:  # Allows running as App.main:app from project root
    from config import UPLOAD_DIR, VIDEO_EXTENSIONS
    from services.full_sentence_video_service import predict_sentence_from_video
    from services.nlp_refiner import refine_to_egyptian


router = APIRouter()


def save_upload_file(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename or "").suffix.lower()

    if suffix not in VIDEO_EXTENSIONS and suffix != ".npz":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    saved_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return saved_path


@router.post("/sign-to-text/sentence-video")
async def sign_to_text_sentence_video(
    file: UploadFile = File(...),
    use_nlp: bool = Form(True),
    window_seconds: float = Form(2.2),
    stride_seconds: float = Form(1.1),
    min_confidence: float = Form(0.35),
    min_margin: float = Form(0.08),
    max_segments: int = Form(40),
    keep_debug_segments: bool = Form(False),
    req_id_form: Optional[str] = Form(default=None, alias="req_id"),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
):
    """
    Receive one full video that may contain multiple signs/words.
    The backend splits the video into overlapping windows and predicts each window.
    """
    req_id = (
        x_request_id.strip()
        if x_request_id and x_request_id.strip()
        else req_id_form.strip()
        if req_id_form and req_id_form.strip()
        else uuid4().hex
    )

    try:
        saved_path = save_upload_file(file)

        result = predict_sentence_from_video(
            video_path=saved_path,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_confidence=min_confidence,
            min_margin=min_margin,
            max_segments=max_segments,
            keep_debug_segments=keep_debug_segments,
        )

        raw_sentence = result["raw_sentence"]
        final_sentence = refine_to_egyptian(raw_sentence) if use_nlp else raw_sentence

        return {
            "status": "success",
            "mode": "sentence_video_word_level",
            "req_id": req_id,
            "filename": file.filename,
            "raw_words": result["raw_words"],
            "raw_sentence": raw_sentence,
            "final_sentence": final_sentence,
            "video_info": result["video_info"],
            "settings": {
                "window_seconds": result["window_seconds"],
                "stride_seconds": result["stride_seconds"],
                "min_confidence": result["min_confidence"],
                "min_margin": result["min_margin"],
                "max_segments": max_segments,
            },
            "segments": result["segments"],
            "debug_segments_dir": result["debug_segments_dir"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "req_id": req_id,
            "filename": file.filename,
            "error": repr(e),
        })
