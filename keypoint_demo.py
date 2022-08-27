import tensorflow.compat.v1 as tf
import cv2
import time
import argparse
import numpy as np

import math

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def get_angle(vec1, vec2):
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return np.degrees(math.acos(cos))

def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0

        leftWrist = [-1, -1]
        rightWrist = [-1, -1]
        leftElbow = [-1, -1]
        rightElbow = [-1, -1]
        leftShoulder = [-1, -1]
        rightShoulder = [-1, -1]

        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            # keypoint_coords->フレームに映った人体のリスト
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            # posenetで見つけたkeypointsを画面に上書きする
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            
            # keypointを表示させない場合はoverlay_imageをdisplay_imageに差し替える
            # overlay_image = display_image

            # フレーム毎に人体のパーツの座標を出力する
            # フレーム数を出力する
            print(frame_count)
            for obj in keypoint_coords:
            # ゼロ行列ではない場合人体の座標であるとして出力する
                if np.all(obj != 0):
                    # 1人ごとに点線を出力する
                    print("-" * 30)
                    # キーポイントを出力する
                    # print(obj)
                    # 手首のピクセル上の移動距離を出力する
                    if np.all(leftWrist != -1):
                        print(np.linalg.norm(obj[9] - leftWrist))
                    if np.all(rightWrist != -1):
                        print(np.linalg.norm(obj[10] - rightWrist))
                    # 肩の座標を更新する
                    leftShoulder = obj[5]
                    rightShoulder = obj[6]
                    # 肘の座標を更新する
                    leftElbow = obj[7]
                    rightElbow = obj[8]
                    # 手首の座標を更新する
                    leftWrist = obj[9]
                    rightWrist = obj[10]

                    # 左肘の角度を計算する
                    print("left ang" + str(get_angle(leftShoulder - leftElbow, leftWrist - leftElbow)))

                    # 右肘の角度を計算する
                    print("right ang" + str(get_angle(rightShoulder - rightElbow, rightWrist - rightElbow)))
                    
            # posenetで処理した画像を映し出す
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            # qキーが押されたら処理を中止する
            if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                break

        # ここから先はバグのため到達不可
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
