from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import requests
from io import BytesIO
import os

from services.animation_generator import AnimationGenerator

video_router = APIRouter()
animator = AnimationGenerator()


@video_router.get("/video/{request_id}")
def get_video(request_id: str):

    pose_url = f"https://res.cloudinary.com/ddzdrjfpb/raw/upload/v1780350750/generated_pose/{request_id}.pose"

    response = requests.get(pose_url, timeout=30)

    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Pose not found")

    pose_buffer = BytesIO(response.content)

    video_path = animator.generate(pose_buffer)

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=500, detail="Video was not generated")

    def iterfile():
        try:
            with open(video_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    yield chunk
        finally:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except Exception as e:
                print("Video cleanup error:", e)

    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "X-Request-Id": request_id,
            "Content-Disposition": f'inline; filename="{request_id}.mp4"'
        }
    )