from spoken_to_signed.gloss_to_pose import concatenate_poses


class PoseSmoother:

    def smooth(self, poses):

        if len(poses) == 0:
            return None

        stitched_pose = concatenate_poses(poses)

        return stitched_pose