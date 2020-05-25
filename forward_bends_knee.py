from threading import Thread
from time import sleep

import posenet
import webcam_run


def return_dest_part(hard):
    return ("leftAnkle", "rightAnkle") if hard else ("leftKnee", "rightKnee")


def forward_bends_knee(is_hard=False):
    outputs = webcam_run.main('Forward bends')

    count = 0
    startEx = False
    dest_left, dest_right = return_dest_part(is_hard)
    while True:
        try:
            pose_scores, keypoint_scores, kp_coords = next(outputs)
            for pose in range(len(pose_scores)):
                if pose_scores[pose] == 0.:
                    break

                left_dest = posenet.PART_NAMES.index(dest_left)
                right_dest = posenet.PART_NAMES.index(dest_right)
                left_wrist = posenet.PART_NAMES.index("leftWrist")
                right_wrist = posenet.PART_NAMES.index("rightWrist")
                left_hip = posenet.PART_NAMES.index("leftHip")
                right_hip = posenet.PART_NAMES.index("rightHip")
                # print("lololo")
                # print(kp_coords[pose, left_knee, :])
                # print(kp_coords[pose, right_knee, :])
                # print(kp_coords[pose, left_wrist, :])
                # print(kp_coords[pose, right_wrist, :])
                # print(kp_coords[pose, left_hip, :])
                # print(kp_coords[pose, right_hip, :])
                if keypoint_scores[pose, left_hip] > 0.5 and keypoint_scores[pose, right_hip] > 0.5 \
                        and keypoint_scores[pose, left_wrist] > 0.5 and keypoint_scores[pose, right_wrist] > 0.5:

                    if abs(kp_coords[pose, left_hip, :][1] - kp_coords[pose, left_wrist, :][1]) < 70 and abs(
                            kp_coords[pose, right_hip, :][1] - kp_coords[pose, right_wrist, :][1]) < 70:
                        print("zaczynamy powtorzenie")
                        startEx = True

                if keypoint_scores[pose, left_dest] > 0.4 and keypoint_scores[pose, right_dest] > 0.4 \
                        and keypoint_scores[pose, left_wrist] > 0.4 and keypoint_scores[pose, right_wrist] > 0.4 and startEx:

                    print(kp_coords[pose, left_dest, :][0] - kp_coords[pose, left_wrist, :][0])
                    print(kp_coords[pose, right_dest, :][0] - kp_coords[pose, right_wrist, :][0])
                    if abs(
                            kp_coords[pose, left_dest, :][0] - kp_coords[pose, left_wrist, :][0]) < 50 and abs(
                        kp_coords[pose, right_dest, :][0] - kp_coords[pose, right_wrist, :][0]) <50:
                        startEx = False
                        count += 1
                        print("TEST2")

        except StopIteration:
            break

    print(f"Bends:{count}")


if __name__ == "__main__":
    forward_bends_knee()
