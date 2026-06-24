import os
import uuid
import cloudinary.uploader

from configs import cloudinary_config
from fastapi import APIRouter, UploadFile, File, Form
from services.arabic_normalizer import ArabicNormalizer
from services.pose_retriever import PoseRetriever
from services.pose_smoother import PoseSmoother
from services.animation_generator import AnimationGenerator
from services.speech_to_text import SpeechToTextSR
from fastapi.responses import StreamingResponse
from io import BytesIO

speech_router = APIRouter()

poses_dir = "poses.json"

normalizer = ArabicNormalizer()
retriever = PoseRetriever(poses_dir)
smoother = PoseSmoother()
animator = AnimationGenerator()
stt = SpeechToTextSR()

@speech_router.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...), request_id: str | None = Form(None)):
    
    request_id = request_id or str(uuid.uuid4())
    
    temp_audio = f"temp_{audio.filename}"

    with open(temp_audio, "wb") as f:
        f.write(await audio.read())
        
    sentence = stt.transcribe_any(temp_audio)
    tokens = normalizer.tokenize(sentence)
    
    try:
        os.remove(temp_audio)
    except:
        pass

    if not tokens:
        return {
            "success": False,
            "message": "No matching tokens found"
        }

    poses = retriever.retrieve(tokens)

    if not poses:
        return {
            "success": False,
            "message": "No pose files found"
        }

    stitched_pose = smoother.smooth(poses)

    if stitched_pose is None:
        return {
            "success": False,
            "message": "Pose stitching failed"
        }
    
    buffer = BytesIO()

    stitched_pose.write(buffer)

    buffer.seek(0)
    
    result = cloudinary.uploader.upload(
        buffer,
        resource_type = "raw",
        folder = "generated_pose",
        public_id=request_id,
        use_filename=False,
        overwrite=True,
        format="pose"
    )
    
    generated_pose_url = result["secure_url"]
    
    
    print("Uploaded:", generated_pose_url)

    
    return {
        "success": True,
        "request_id": request_id,
        "pose_URL": generated_pose_url,
        "sentence": sentence,
        "tokens": tokens
    }