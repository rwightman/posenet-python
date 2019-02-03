import tensorflow as tf
import numpy as np
import cv2
import time
import argparse
import os

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        max_pose_detections = 10
        pose_scores = np.zeros(max_pose_detections, dtype=np.float32)
        pose_keypoint_scores = np.zeros((max_pose_detections, posenet.NUM_KEYPOINTS), dtype=np.float32)
        pose_keypoint_coords = np.zeros((max_pose_detections, posenet.NUM_KEYPOINTS, 2), dtype=np.float32)

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_count = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                pose_scores,
                pose_keypoint_scores,
                pose_keypoint_coords,
                output_stride=output_stride,
                min_pose_score=0.25)

            pose_keypoint_coords *= output_scale

            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, pose_keypoint_scores, pose_keypoint_coords, pose_count,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(pose_count):
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(pose_keypoint_scores[pi, :], pose_keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
