""" Program counts hands up"""
import time
import copy
import posenet
import webcam_run
import numpy as np
import math

from count_exercises import get_pose


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
    return abs(np.degrees(angle) - 90) < 20


def hands_up():
    outputs = webcam_run.main('Hands up')

    count = 0
    startEx = False
    while True:
        try:
            pose_scores, keypoint_scores, kp_coords = next(outputs)
            for pose in range(len(pose_scores)):
                if pose_scores[pose] == 0.:
                    break
                left_shoulder = posenet.PART_NAMES.index("leftShoulder")
                right_shoulder = posenet.PART_NAMES.index("rightShoulder")
                left_elbow = posenet.PART_NAMES.index("leftElbow")
                right_elbow = posenet.PART_NAMES.index("rightElbow")
                left_wrist = posenet.PART_NAMES.index("leftWrist")
                right_wrist = posenet.PART_NAMES.index("rightWrist")
                left_hip = posenet.PART_NAMES.index("leftHip")
                right_hip = posenet.PART_NAMES.index("rightHip")

                if keypoint_scores[pose, left_shoulder] > 0.5 and keypoint_scores[pose, left_elbow] > 0.5 \
                        and keypoint_scores[pose, left_wrist] > 0.5 and keypoint_scores[pose, left_hip] > 0.5:
                    print("TEST")
                    # print(kp_coords[pose, left_shoulder, :])
                    # print(kp_coords[pose, left_elbow, :])
                    # print(kp_coords[pose, left_wrist, :])
                    # print(kp_coords[pose, left_hip, :])
                    if not startEx and abs(
                            kp_coords[pose, left_shoulder, :][1] - kp_coords[pose, left_elbow, :][1]) < 30 and abs(
                            kp_coords[pose, left_elbow, :][1] - kp_coords[pose, left_wrist, :][1]) < 30:
                        # if not startEx and abs(kp_coords[pose, left_hip, :][0]-kp_coords[pose,left_elbow, :][0]) <30 :
                        print("start")
                        startEx = True
                        is90(kp_coords[pose, left_hip, :], kp_coords[pose, left_shoulder, :],
                             kp_coords[pose, left_wrist, :])
                if startEx:

                    if isLine2([kp_coords[pose, left_shoulder, :][1], kp_coords[pose, left_shoulder, :][0]],
                               [kp_coords[pose, left_elbow, :][1], kp_coords[pose, left_elbow, :][0]],
                               [kp_coords[pose, left_wrist, :][1], kp_coords[pose, left_wrist, :][0]]):
                        if is90(kp_coords[pose, left_hip, :], kp_coords[pose, left_shoulder, :],
                                kp_coords[pose, left_wrist, :]):
                            print("mamy kąt prosty")
                            count += 1
                            # pose_scores, keypoint_scores, kp_coords = get_pose(
                            #     output_stride, cap, str(count), sess, model_outputs)
                            startEx = False

        except StopIteration:
            break

    print("Hands up: ", count)


if __name__ == "__main__":
    hands_up()
