import argparse
import socket
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=4219)
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", args.port))
server.listen()

def detect_body(model_outputs, output_stride, sess, img):
    input_image, display_image, output_scale = posenet.read_cvimg(
        img, scale_factor=args.scale_factor, output_stride=output_stride)
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
    cv2.imshow('body', overlay_image)
    cv2.waitKey(1)


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        while True:
            client, _ = server.accept()
            img = bytearray()
            while True:
                data = client.recv(1024)
                if not data:
                    break
                img.extend(data)
            client.close()
            img = np.frombuffer(img, dtype=np.uint8).reshape(args.cam_height,
             args.cam_width, -1)
            detect_body(model_outputs, output_stride, sess, img)

if __name__ == "__main__":
    main()