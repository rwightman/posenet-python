import posenet
import webcam_run


def forward_bends_knee():
    outputs = webcam_run.main('Forward bends')

    count = 0
    startEx = False
    print("[BENDS] Starting...")
    while True:
        try:
            pose_scores, keypoint_scores, kp_coords = next(outputs)
            for pose in range(len(pose_scores)):
                if pose_scores[pose] == 0.:
                    break

                left_dest = posenet.PART_NAMES.index("leftKnee")
                right_dest = posenet.PART_NAMES.index("rightKnee")
                left_wrist = posenet.PART_NAMES.index("leftWrist")
                right_wrist = posenet.PART_NAMES.index("rightWrist")
                left_hip = posenet.PART_NAMES.index("leftHip")
                right_hip = posenet.PART_NAMES.index("rightHip")

                if keypoint_scores[pose, left_hip] > 0.5 and keypoint_scores[pose, right_hip] > 0.5 \
                        and keypoint_scores[pose, left_wrist] > 0.5 and keypoint_scores[pose, right_wrist] > 0.5:

                    if abs(kp_coords[pose, left_hip, :][1] - kp_coords[pose, left_wrist, :][1]) < 70 and abs(
                            kp_coords[pose, right_hip, :][1] - kp_coords[pose, right_wrist, :][1]) < 70:
                        print("Start")
                        startEx = True

                if keypoint_scores[pose, left_dest] > 0.4 and keypoint_scores[pose, right_dest] > 0.4 \
                        and keypoint_scores[pose, left_wrist] > 0.4 and keypoint_scores[pose, right_wrist] > 0.4 \
                        and startEx:

                    if abs(kp_coords[pose, left_dest, :][0] - kp_coords[pose, left_wrist, :][0]) < 50 and abs(
                            kp_coords[pose, right_dest, :][0] - kp_coords[pose, right_wrist, :][0]) < 50:
                        startEx = False
                        count += 1
                        print(f"End, number: {count}")

        except StopIteration:
            break

    print(f"Bends:{count}")


if __name__ == "__main__":
    forward_bends_knee()
