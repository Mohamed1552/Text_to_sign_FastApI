from pathlib import Path
from typing import Optional
from uuid import uuid4
import shutil

from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException

from config import UPLOAD_DIR, VIDEO_EXTENSIONS
from services.full_sentence_video_service import predict_full_sentence_video
from services.nlp_refiner import refine_to_egyptian


router = APIRouter()

# Do not require config.TEMP_DIR, to avoid breaking your old config.py.
UPLOAD_DIR = Path(UPLOAD_DIR)
TEMP_DIR = UPLOAD_DIR.parent if UPLOAD_DIR.name.lower() == "uploads" else UPLOAD_DIR / "temp"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def save_full_sentence_upload(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename or "").suffix.lower()

    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type for full sentence video: {suffix}",
        )

    saved_path = UPLOAD_DIR / f"full_sentence_{uuid4().hex}{suffix}"

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return saved_path


@router.post("/sign-to-text/full-sentence-video")
async def full_sentence_video_endpoint(
    file: UploadFile = File(...),
    use_nlp: bool = Form(True),
    # Faster defaults than the first version.
    window_seconds: float = Form(3.0),
    stride_seconds: float = Form(2.5),
    min_confidence: float = Form(0.30),
    min_margin: float = Form(0.08),
    min_time_gap: float = Form(4.0),
    max_segments: int = Form(10),
    debug: bool = Form(False),
    req_id_form: Optional[str] = Form(default=None, alias="req_id"),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
):
    req_id = (
        x_request_id.strip()
        if x_request_id and x_request_id.strip()
        else req_id_form.strip()
        if req_id_form and req_id_form.strip()
        else uuid4().hex
    )

    try:
        saved_path = save_full_sentence_upload(file)

        result = predict_full_sentence_video(
            video_path=saved_path,
            temp_dir=TEMP_DIR,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            min_confidence=min_confidence,
            min_margin=min_margin,
            min_time_gap=min_time_gap,
            max_segments=max_segments,
            debug=debug,
        )

        raw_words = result["raw_words"]
        raw_sentence = " - ".join(raw_words)
        final_sentence = refine_to_egyptian(raw_sentence) if use_nlp else raw_sentence

        return {
            "status": "success",
            "mode": "full_sentence_video_word_level_v2",
            "req_id": req_id,
            "filename": file.filename,
            "raw_words": raw_words,
            "raw_sentence": raw_sentence,
            "final_sentence": final_sentence,
            # "video_info": result["video_info"],
            # "settings": result["settings"],
            # "segments": result["segments"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "req_id": req_id,
            "filename": file.filename,
            "error": repr(e),
        })
