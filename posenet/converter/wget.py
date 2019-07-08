import urllib.request
import posixpath
import json
import zlib
import os

from posenet.converter.config import load_config

CFG = load_config()
GOOGLE_CLOUD_STORAGE_DIR = CFG['GOOGLE_CLOUD_STORAGE_DIR']
CHECKPOINTS = CFG['checkpoints']
CHK = CFG['chk']


def download_file(checkpoint, filename, base_dir):
    output_path = os.path.join(base_dir, checkpoint, filename)
    url = posixpath.join(GOOGLE_CLOUD_STORAGE_DIR, checkpoint, filename)
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    if response.info().get('Content-Encoding') == 'gzip':
        data = zlib.decompress(response.read(), zlib.MAX_WBITS | 32)
    else:
        # this path not tested since gzip encoding default on google server
        # may need additional encoding/text handling if hit in the future
        data = response.read()
    with open(output_path, 'wb') as f:
        f.write(data)


def download(checkpoint, base_dir='./weights/'):
    save_dir = os.path.join(base_dir, checkpoint)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    download_file(checkpoint, 'manifest.json', base_dir)
    with open(os.path.join(save_dir, 'manifest.json'), 'r') as f:
        json_dict = json.load(f)

    for x in json_dict:
        filename = json_dict[x]['filename']
        print('Downloading', filename)
        download_file(checkpoint, filename, base_dir)


def main():
    checkpoint = CHECKPOINTS[CHK]
    download(checkpoint)


if __name__ == "__main__":
    main()
