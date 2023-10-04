import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import open3d as o3d
import trimesh
sys.path.append('.')
from lib.utils import data_utils
import torch
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_grid_dir', type=str, default='/home/linhaotong/workspace/im4d/data/htcode/result/0013_01/im4d/grid/sample')
    parser.add_argument('--input_mesh_dir', type=str, default='/home/linhaotong/workspace/im4d/data/htcode/result/0013_01/im4d/0013_01_demo/pcd_1003/meshes')
    parser.add_argument('--output_grid_dir', type=str, default='/home/linhaotong/workspace/im4d/data/htcode/result/0013_01/im4d/grid/pcd')
    parser.add_argument('--padding_grid', type=bool, default=False)
    parser.add_argument('--conv_size', type=int, default=3)
    args = parser.parse_args()
    return args

def main(args):
    grid_size = (128, 256, 128)
    padding_size = 0.1
    bounds = np.load(join(args.input_grid_dir, 'bounds.npz'))['arr_0']
    binarys = np.load(join(args.input_grid_dir, 'binarys.npz'))['arr_0']
    
    new_bounds = []
    new_binarys = []
    for frame_id in tqdm(range(len(bounds))):
        bound = bounds[frame_id]
        if not os.path.exists(join(args.input_mesh_dir, '{:06d}.ply'.format(frame_id))) and frame_id not in [0, 20, 40, 60, 80, 100, 120, 140]:
            new_bounds.append(bounds[frame_id])
            new_binarys.append(binarys[frame_id])
            continue
        pcd_denoise = data_utils.read_and_denoise(join(args.input_mesh_dir, '{:06d}.ply'.format(frame_id)), vox_size=0.001)
        points = np.asarray(pcd_denoise.points)
        
        # cropping points out of bound
        points = points[(points[:, 0] > bound[0, 0]) & (points[:, 0] < bound[1, 0])]
        points = points[(points[:, 1] > bound[0, 1]) & (points[:, 1] < bound[1, 1])]
        points = points[(points[:, 2] > bound[0, 2]) & (points[:, 2] < bound[1, 2])]
        
        mesh = trimesh.PointCloud(points)
        mesh_path = join(args.output_grid_dir, 'meshes', '{:06d}.ply'.format(frame_id))
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        print(mesh_path)
        mesh.export(mesh_path)
        
        # min_bound = points.min(axis=0) - padding_size
        # max_bound = points.max(axis=0) + padding_size
        min_bound = bound[0]
        max_bound = bound[1]
        voxel_size = (max_bound - min_bound) / np.array(grid_size)
        indices = ((points - min_bound) // voxel_size).astype(int)
        voxel_grid = np.zeros(grid_size, dtype=np.uint8)
        voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        
        if args.padding_grid:
            conv_size = args.conv_size
            binary = torch.tensor(voxel_grid).cuda()
            structuring_element = torch.ones(1, 1, conv_size, conv_size, conv_size, device=binary.device)
            dilated_tensor = F.conv3d(binary[None, None].float(), structuring_element.float(), padding=conv_size//2) > 0
            voxel_grid = (dilated_tensor.cpu().numpy()).astype(np.uint8)[0, 0]
       
        print(frame_id, voxel_size, binarys[frame_id].sum(), voxel_grid.sum())
        
        new_bounds.append(np.stack([min_bound, max_bound]).astype(np.float32))
        new_binarys.append(voxel_grid.astype(np.bool))
    os.makedirs(args.output_grid_dir, exist_ok=True)
    np.savez_compressed(join(args.output_grid_dir, 'bounds.npz'), np.array(new_bounds))
    np.savez_compressed(join(args.output_grid_dir, 'binarys.npz'), np.array(new_binarys))
        

if __name__ == '__main__':
    args = parse_args()
    main(args)

