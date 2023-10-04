import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from trdparty.colmap.read_write_model import read_model, read_points3d_binary
import trimesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--thresh', type=int, default=3)
    parser.add_argument('--use_roimask', type=bool, default=True)
    args = parser.parse_args()
    return args

def main(args):
    input_dir = args.input
    output_path = args.output
    # cameras, images, points3D = read_model(input_dir)
    points3D = read_points3d_binary(join(input_dir, 'points3D.bin'))
    points = np.array([points3D[k].xyz for k in points3D if len(points3D[k].image_ids) > args.thresh ])
    rgbs = np.array([points3D[k].rgb for k in points3D if len(points3D[k].image_ids) > args.thresh])
    if args.use_roimask:
        roi = np.stack([np.percentile(points, 0.1, axis=0), np.percentile(points, 99.9, axis=0)])
        mask = np.logical_and(points > roi[:1], points < roi[1:])
        mask = mask.sum(axis=-1) == 3
        points = points[mask]
        rgbs = rgbs[mask]
    print('Number of points: ', len(points))
    pcd = trimesh.PointCloud(vertices=points, colors=rgbs)
    pcd.export(output_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)

