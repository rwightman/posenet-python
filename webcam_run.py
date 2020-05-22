import tensorflow as tf
import cv2
import time
import posenet

# (0-3) or integer depth multiplier (50, 75, 100, 101)
MODEL = 101
CAM_ID = 0


def main(title):
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(MODEL, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture(CAM_ID)
        if not cap.isOpened():
            raise Exception("Could not open video device")
        # resolution of video 1080p = 1920x1080 =>
        cap.set(3, 1920)
        cap.set(4, 1080)

        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=0.7125, output_stride=output_stride)

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

            yield (pose_scores, keypoint_scores, keypoint_coords)

            cv2.imshow(title, overlay_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main('posenet')