from pose_format.pose_visualizer import PoseVisualizer
from pose_format import Pose
import tempfile
import os
import subprocess
import imageio_ffmpeg


class AnimationGenerator:

    def generate(self, pose):
        """
        Generate a browser-compatible MP4 video.

        PoseVisualizer/OpenCV may create MP4 using the `mp4v` codec.
        Some browsers, Windows players, and Postman previews do not support it.
        So we first generate a temporary OpenCV MP4, then transcode it to H.264
        with yuv420p pixel format using the bundled imageio-ffmpeg binary.
        """
        if hasattr(pose, "read"):
            pose.seek(0)
            pose = Pose.read(pose)

        elif isinstance(pose, str):
            with open(pose, "rb") as f:
                pose = Pose.read(f)

        visualizer = PoseVisualizer(pose)

        raw_video = tempfile.NamedTemporaryFile(
            suffix="_raw.mp4",
            delete=False
        )
        raw_video.close()

        final_video = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        )
        final_video.close()

        try:
            # 1) Generate initial video from pose_format / OpenCV
            visualizer.save_video(
                raw_video.name,
                visualizer.draw()
            )

            if not os.path.exists(raw_video.name) or os.path.getsize(raw_video.name) == 0:
                raise RuntimeError("PoseVisualizer generated an empty raw video file")

            # 2) Convert to browser-compatible H.264 MP4
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", raw_video.name,
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                final_video.name
            ]

            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    "FFmpeg H.264 conversion failed: " + completed.stderr[-2000:]
                )

            if not os.path.exists(final_video.name) or os.path.getsize(final_video.name) == 0:
                raise RuntimeError("FFmpeg produced an empty H.264 video file")

            return final_video.name

        finally:
            try:
                if os.path.exists(raw_video.name):
                    os.remove(raw_video.name)
            except Exception:
                pass
