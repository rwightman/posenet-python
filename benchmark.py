import tensorflow as tf
import time
import argparse
import os

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--num_images', type=int, default=1000)
args = parser.parse_args()


def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        num_images = args.num_images

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
        if len(filenames) > num_images:
            filenames = filenames[:num_images]

        images = {f: posenet.read_imgfile(f, 1.0, output_stride)[0] for f in filenames}

        start = time.time()
        for i in range(num_images):
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': images[filenames[i % len(filenames)]]}
            )

            output = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        print('Average FPS:', num_images / (time.time() - start))


if __name__ == "__main__":
    main()
