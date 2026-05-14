from fastapi import APIRouter
import os
import uuid
from config import cloudinary_config
import cloudinary.uploader

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

    # ==============================
    # Create unique filenames
    # ==============================

    unique_id = str(uuid.uuid4())

    output_dir = "static\output"
    os.makedirs(output_dir, exist_ok=True)

    pose_output = os.path.join(
        output_dir,
        f"{unique_id}.pose"
    )

    video_output = os.path.join(
        output_dir,
        f"{unique_id}.mp4"
    )

    # ==============================
    # Save pose
    # ==============================

    with open(pose_output, "wb") as f:
        stitched_pose.write(f)

    result = cloudinary.uploader.upload(
        pose_output,
        resource_type = "raw",
        folder = "generated_pose",
        use_filename = True,
        unique_filename = True
    )
    
    generated_pose_url = result["secure_url"]
    # ==============================
    # Generate video
    # ==============================

    animator.generate(stitched_pose, video_output)
    
    print("Uploaded:", generated_pose_url)
    
    return {
        "success": True,
        "pose_file": pose_output,
        "video_file": video_output,
        "pose_URL": generated_pose_url,
        "tokens": tokens
    }