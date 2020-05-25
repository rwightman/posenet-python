import time
import tensorflow as tf
import posenet
import webcam_run
import time
import argparse
import copy
import numpy as np
import tensorflow as tf
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


def check_eyes(keypoint_scores, pose, eye):
    part_eye = posenet.PART_NAMES.index(eye)
    print(keypoint_scores[pose, part_eye])
    return keypoint_scores[pose, part_eye] < 0.4



def head_ex():
    outputs = webcam_run.main('Head spinning')

    count_left = 0
    count_right = 0
    startEx = False
    while True:
        try:
            pose_scores, keypoint_scores, kp_coords = next(outputs)
            for pose in range(len(pose_scores)):
                if pose_scores[pose] == 0.:
                    break
                left_ear = posenet.PART_NAMES.index("leftEar")
                right_ear = posenet.PART_NAMES.index("rightEar")



                if keypoint_scores[pose, left_ear] > 0.5 and keypoint_scores[pose, right_ear] > 0.5:
                    print("start")
                    startEx = True
                if keypoint_scores[pose, left_ear] < 0.5 < keypoint_scores[pose, right_ear] and check_eyes(keypoint_scores, pose, "rightEye") and startEx:
                    print(f"Turning left: {count_left}")
                    count_left += 1
                    startEx = False
                if keypoint_scores[pose, right_ear] < 0.5 < keypoint_scores[pose, left_ear] and check_eyes(keypoint_scores, pose, "leftEye") and startEx:
                    print(f"Turning left: {count_right}")
                    count_right += 1
                    startEx = False

        except StopIteration:
            break

    print(f"Turn right:{count_right} Turn left:{count_left} ")


if __name__ == "__main__":
    head_ex()
