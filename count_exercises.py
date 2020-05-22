"""Programme counts different exercises"""
import time
import argparse
import copy
import numpy as np
import tensorflow as tf
import cv2

import posenet


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model', type=int, default=101)
PARSER.add_argument('--cam_id', type=int, default=0)
PARSER.add_argument('--cam_width', type=int, default=1280)
PARSER.add_argument('--cam_height', type=int, default=720)
PARSER.add_argument('--scale_factor', type=float, default=0.7125)
PARSER.add_argument(
    '--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
ARGS = PARSER.parse_args()


def find_initial(output_stride, cap, sess, model_outputs):
    """While person stays still gets its coordinates and return average."""

    glob_count = 0
    while glob_count < 40:
        count = 1
        pose_scores_start, _, kp_coords_start = get_pose(
            output_stride, cap, "Stay still", sess, model_outputs)
        for pose, kp_coord_start in enumerate(kp_coords_start):
            if pose_scores_start[pose] != 0.:
                glob_count += 1
                try:
                    kp_coords_start_av = kp_coords_start_av + kp_coord_start
                except NameError:
                    kp_coords_start_av = copy.deepcopy(kp_coord_start)
                else:
                    count += 1
        if count != 1:
            kp_coords_start_av = np.divide(kp_coords_start_av, count)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return kp_coords_start_av


def get_pose(output_stride, cap, legend, sess, model_outputs):
    """Gets pose's coordinates, draws lines ans prints text."""

    input_image, display_image, output_scale = posenet.read_cap(
        cap, scale_factor=ARGS.scale_factor, output_stride=output_stride)

    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
        model_outputs,
        feed_dict={'image:0': input_image}
    )

    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=10,
        min_pose_score=0.15)

    keypoint_coords *= output_scale

    overlay_image = posenet.draw_skel_and_kp(
        display_image, pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.15, min_part_score=0.1)
    if legend:
        cv2.putText(
            overlay_image, legend, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 20)
    cv2.imshow('posenet', overlay_image)
    return (pose_scores, keypoint_scores, keypoint_coords)


def main(amount, exercise):
    """ Wait 3 sec for person to stand in a right position, find initial position,
        count exercises.
    """

    time.sleep(3)
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(ARGS.model, sess)
        output_stride = model_cfg['output_stride']
        if ARGS.file is not None:
            cap = cv2.VideoCapture(ARGS.file)
        else:
            cap = cv2.VideoCapture(ARGS.cam_id)
        cap.set(3, ARGS.cam_width)
        cap.set(4, ARGS.cam_height)

        kp_coords_start_av = find_initial(output_stride, cap, sess, model_outputs)
        #print("kp_coords_start_av: ", kp_coords_start_av)
        compare_val = abs(kp_coords_start_av[0, 0] - kp_coords_start_av[-1, 0])*0.03
        #print("compare_val: ", compare_val)

        count = 0
        down = False
        amount = int(amount)
        if exercise == 'squart':
            while count < amount:
                count, down = count_squats(
                    output_stride, cap, sess, model_outputs,
                    count, kp_coords_start_av, compare_val, down)
                #print("COUNT: ", count)
                #print("DOWN: ", down)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif exercise == 'arms':
            pass # TO DO count arms
        get_pose(output_stride, cap, "GREAT!", sess, model_outputs)


def count_squats(output_stride, cap, sess, model_outputs,
                 count, kp_coords_start_av, compare_val, down):
    """Gets pose's coordinates, checks if difference in knees height is big enough,
    if yes - adds squat.
    """

    pose_scores, keypoint_scores, kp_coords = get_pose(
        output_stride, cap, str(count), sess, model_outputs)
    for pose in range(len(pose_scores)):
        if pose_scores[pose] == 0.:
            break
        #for ki, (s, c) in enumerate(zip(keypoint_scores[pose, :], kp_coords[pose, :, :])):
        #    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
        l_knee_in = posenet.PART_NAMES.index("leftKnee")
        r_knee_in = posenet.PART_NAMES.index("rightKnee")
        if keypoint_scores[pose, l_knee_in] > 0.7:
            diff = kp_coords[pose, l_knee_in, :] - kp_coords_start_av[l_knee_in, :]
        elif keypoint_scores[pose, r_knee_in] > 0.7:
            diff = kp_coords[pose, r_knee_in, :] - kp_coords_start_av[r_knee_in, :]
        else:
            break
        #print("DIFF: ", diff)
        if diff[0] > compare_val and not down:
            down = True
        elif diff[0] < compare_val and diff[0] > 0 and down:
            down = False
            count += 1
    return (count, down)
