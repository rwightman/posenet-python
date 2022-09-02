import argparse
import socket
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import posenet
import requests
import notification
import time
import redis

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=4219)
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--moving_threshold', type=int, default=200)
parser.add_argument('--redis_host', type=str, default='localhost')
parser.add_argument('--redis_port', type=int, default='6379')
parser.add_argument('--webhook_endpoint', type=str)
args = parser.parse_args()
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", args.port))
server.listen()

def detect_body(model_outputs, output_stride, sess, frm):
    input_image, display_image, output_scale = posenet.read_cvimg(
        frm, scale_factor=args.scale_factor, output_stride=output_stride)
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
    overlay_image = display_image * posenet.fill_kp(
        np.zeros_like(display_image), pose_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.15, min_part_score=0.1)
    return overlay_image

def detect_motion(frm, prv_frm):
    if prv_frm is None:
        return np.zeros_like(frm)
    cv2.accumulateWeighted(frm, prv_frm, 0.6)
    diff = cv2.absdiff(frm, cv2.convertScaleAbs(prv_frm))
    thresh = (cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]).copy().astype("uint8")
    return thresh / 255 * frm

def main():
    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    prv_frm = {}
    current_target = ''
    count = 0
    last_access = 0.0
    inst = ''
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        while True:
            print(count)
            if time.time() - last_access > 1.0:
                inst = redis_client.get('instruction')
                if current_target != redis_client.get("current_target"):
                    count = 0
                    current_target = redis_client.get("current_target")
                last_access = time.time()
            client, addr = server.accept()
            img = bytearray()
            while True:
                data = client.recv(1024)
                if not data:
                    break
                img.extend(data)
            client.close()
            frm = np.frombuffer(img, dtype=np.uint8).reshape(args.cam_height,
             args.cam_width, -1)
            cv2.imshow('original', frm)
            body_frm = detect_body(model_outputs, output_stride, sess, frm)
            cv2.imshow('wrist detect', body_frm)
            gray = cv2.cvtColor(body_frm, cv2.COLOR_BGR2GRAY)
            motion_frm = detect_motion(gray, prv_frm.get(addr[0]))
            cv2.imshow('motion detect', motion_frm)
            cv2.waitKey(1)
            prv_frm[addr[0]] = gray.copy().astype("float")
            if inst != 'observe':
                continue
            if motion_frm.mean() < 10:
                count += 1
            if count >= 200:
                notification.stalemate_notification(current_target, args.webhook_endpoint)
                count = 0

if __name__ == "__main__":
    main()