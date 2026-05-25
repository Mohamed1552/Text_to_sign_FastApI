from pose_format.pose_visualizer import PoseVisualizer
import tempfile


class AnimationGenerator:

    def generate(self, pose):

        visualizer = PoseVisualizer(pose)

        temp_video = tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False
        )

        temp_video.close()

        visualizer.save_video(
            temp_video.name,
            visualizer.draw()
        )

        return temp_video.name