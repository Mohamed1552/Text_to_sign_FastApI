from pose_format.pose_visualizer import PoseVisualizer


class AnimationGenerator:

    def generate(self, pose, output_path):

        visualizer = PoseVisualizer(pose)

        visualizer.save_video(output_path, visualizer.draw())

        print("Video saved:", output_path)