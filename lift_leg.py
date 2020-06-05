import time
import tensorflow as tf
import posenet
import webcam_run
import argparse
import copy
import numpy as np
import cv2

from count_exercises import get_pose

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model', type=int, default=101)
PARSER.add_argument('--cam_id', type=int, default=0)
PARSER.add_argument('--cam_width', type=int, default=1280)
PARSER.add_argument('--cam_height', type=int, default=720)
PARSER.add_argument('--scale_factor', type=float, default=0.7125)
PARSER.add_argument(
    '--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
ARGS = PARSER.parse_args()


def isLine2(point1, point2, point3):
    x = [point1[0], point2[0]]
    y = [point1[1], point2[1]]
    coefficients = np.polyfit(x, y, 1)
    return abs((point3[0] * coefficients[0] + coefficients[1]) - point3[1]) < 50


def is90(hip, shoulder, wrist):
    a = np.array(hip)
    b = np.array(shoulder)
    c = np.array(wrist)

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    print(np.degrees(angle))
    return abs(np.degrees(angle) - 55) < 10


def lift_leg(which_side="left"):
    outputs = webcam_run.main('Hands up')

    count = 0
    startEx = False
    while True:
        try:
            pose_scores, keypoint_scores, kp_coords = next(outputs)
            for pose in range(len(pose_scores)):
                if pose_scores[pose] == 0.:
                    break
                if which_side == "left":
                    left_knee = posenet.PART_NAMES.index("leftKnee")
                    right_knee = posenet.PART_NAMES.index("rightKnee")
                    right_hip = posenet.PART_NAMES.index("rightHip")

                else:
                    left_knee = posenet.PART_NAMES.index("rightKnee")
                    right_knee = posenet.PART_NAMES.index("leftKnee")
                    right_hip = posenet.PART_NAMES.index("leftHip")


                if keypoint_scores[pose, left_knee] > 0.5 and keypoint_scores[pose, right_knee] > 0.5 \
                        and keypoint_scores[pose, right_hip] > 0.5:
                    print("TEST")
                    # print(kp_coords[pose, left_shoulder, :])
                    # print(kp_coords[pose, left_elbow, :])
                    # print(kp_coords[pose, left_wrist, :])
                    # print(kp_coords[pose, left_hip, :])
                    if not startEx and abs(
                            kp_coords[pose, left_knee, :][0] - kp_coords[pose, right_knee, :][0]) < 30:
                        # if not startEx and abs(kp_coords[pose, left_hip, :][0]-kp_coords[pose,left_elbow, :][0]) <30 :
                        print("start")
                        startEx = True
                        # is90(kp_coords[pose, left_hip, :], kp_coords[pose, left_shoulder, :],
                        #      kp_coords[pose, left_wrist, :])
                if startEx:

                    # if isLine2([kp_coords[pose, left_shoulder, :][1], kp_coords[pose, left_shoulder, :][0]],
                    #            [kp_coords[pose, left_elbow, :][1], kp_coords[pose, left_elbow, :][0]],
                    #            [kp_coords[pose, left_wrist, :][1], kp_coords[pose, left_wrist, :][0]]):
                    if is90(kp_coords[pose, right_knee, :], kp_coords[pose, right_hip, :],
                            kp_coords[pose, left_knee, :]):

                        print("mamy kąt prosty")
                        count += 1
                        # pose_scores, keypoint_scores, kp_coords = get_pose(
                        #     output_stride, cap, str(count), sess, model_outputs)
                        startEx = False

        except StopIteration:
            break

    print(f"Lifting {which_side} leg: ", count)


if __name__ == "__main__":
    lift_leg()
