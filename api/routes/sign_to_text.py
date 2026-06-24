from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import shutil

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header

from config import UPLOAD_DIR, VIDEO_EXTENSIONS
from services.video_processor import preprocess_video
from services.sign_predictor import predict_sign
from services.nlp_refiner import refine_to_egyptian
router =APIRouter()

def save_upload_file(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS and suffix != ".npz":
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    saved_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return saved_path

@router.post("/sign-to-text")
async def sign_to_text(
    files: List[UploadFile] = File(...),
    use_nlp: bool = Form(True),
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
    predictions = []
    words = []

    for order, upload_file in enumerate(files, start=1):
        try:
            saved_path = save_upload_file(upload_file)
            preprocessed = preprocess_video(saved_path)
            prediction = predict_sign(preprocessed)

            word = prediction["word"]
            if word:
                words.append(word)

            predictions.append({
                "order": order,
                "filename": upload_file.filename,
                "word": word,
                "prediction": prediction,
            })

        except Exception as e:
            # In production, log the traceback server-side.
            raise HTTPException(status_code=500, detail={
                "filename": upload_file.filename,
                "error": repr(e),
            })

    raw_sentence = " - ".join(words)
    final_sentence = refine_to_egyptian(raw_sentence) if use_nlp else raw_sentence

    return {
        "req_id": req_id,
        "status": "success",
        "raw_words": words,
        "raw_sentence": raw_sentence,
        "final_sentence": final_sentence,
        #"predictions": predictions,
    }