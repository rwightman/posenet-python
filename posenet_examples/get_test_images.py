import urllib.request
import os
import argparse

GOOGLE_CLOUD_IMAGE_BUCKET = 'https://storage.googleapis.com/tfjs-models/assets/posenet/'

TEST_IMAGES = [
  'frisbee.jpg',
  'frisbee_2.jpg',
  'backpackman.jpg',
  'boy_doughnut.jpg',
  'soccer.png',
  'with_computer.jpg',
  'snowboard.jpg',
  'person_bench.jpg',
  'skiing.jpg',
  'fire_hydrant.jpg',
  'kyte.jpg',
  'looking_at_computer.jpg',
  'tennis.jpg',
  'tennis_standing.jpg',
  'truck.jpg',
  'on_bus.jpg',
  'tie_with_beer.jpg',
  'baseball.jpg',
  'multi_skiing.jpg',
  'riding_elephant.jpg',
  'skate_park_venice.jpg',
  'skate_park.jpg',
  'tennis_in_crowd.jpg',
  'two_on_bench.jpg',
]

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./images')
args = parser.parse_args()


def main():
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    for f in TEST_IMAGES:
        url = os.path.join(GOOGLE_CLOUD_IMAGE_BUCKET, f)
        print('Downloading %s' % f)
        urllib.request.urlretrieve(url, os.path.join(args.image_dir, f))


if __name__ == "__main__":
    main()
