import os
import argparse
import open3d as o3d
from tqdm import tqdm

import sys
sys.path.append('.')

from lib.utils import data_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to pointcloud directory')
    args = parser.parse_args()
    return args

def main(args):
    dir = args.input
    dir_denoise = os.path.join(os.path.dirname(dir), 'pointcloud_denoise')
    os.makedirs(dir_denoise, exist_ok=True)
    for file in tqdm(os.listdir(dir)):
        if file.endswith('.ply'):
            pcd_denoise = data_utils.read_and_denoise(os.path.join(dir, file), vox_size=0.01)
            o3d.io.write_point_cloud(os.path.join(dir_denoise, file), pcd_denoise)

if __name__ == '__main__':
    args = parse_args()
    main(args)