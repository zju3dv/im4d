import argparse
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    return args

def main(args):
    bbox = np.zeros((2, 3))

    pcd = o3d.io.read_point_cloud(args.input)
    pcd = np.asarray(pcd.points)

    bbox[0] = np.min(pcd, axis=0)
    bbox[1] = np.max(pcd, axis=0)

    print(bbox)

if __name__ == '__main__':
    args = parse_args()
    main(args)

