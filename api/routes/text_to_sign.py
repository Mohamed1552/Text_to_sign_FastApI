import os
import cloudinary.uploader

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from io import BytesIO
from config import cloudinary_config
from services.arabic_normalizer import ArabicNormalizer
from services.pose_retriever import PoseRetriever
from services.pose_smoother import PoseSmoother
from services.animation_generator import AnimationGenerator
from schemas.requests import TextInput




text_router = APIRouter()

poses_dir = "poses.json"

normalizer = ArabicNormalizer()
retriever = PoseRetriever(poses_dir)
smoother = PoseSmoother()
animator = AnimationGenerator()

@text_router.post("/text-to-sign")
def text_to_sign(data: TextInput):

    tokens = normalizer.tokenize(data.sentence)

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


    video_output = animator.generate(
    stitched_pose
    )
   
    buffer = BytesIO()

    stitched_pose.write(buffer)

    buffer.seek(0)
    
    result = cloudinary.uploader.upload(
        buffer,
        resource_type = "raw",
        folder = "generated_pose",
        use_filename = True,
        unique_filename = True
    )
    
    generated_pose_url = result["secure_url"]

    
    def iterfile():

        with open(video_output, "rb") as f:

            while chunk := f.read(1024 * 1024):
                yield chunk

        os.remove(video_output)
    
    print("Uploaded:", generated_pose_url)
    
    
    return StreamingResponse(
    iterfile(),
    media_type="video/mp4",
    headers={
        "X-Pose-URL": generated_pose_url
    }
)