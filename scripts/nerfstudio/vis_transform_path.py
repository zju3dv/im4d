import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

import json
from lib.utils.data_utils import transform2opencv
from lib.utils.vis_utils import vis_point_cloud, pcd_array2o3d
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--transform_name', type=str)
    args = parser.parse_args()
    return args

def transform_points_opengl_to_opencv(points):
    # 创建一个新的数组，因为我们不想修改原始数据
    new_points = points.copy()

    # 在OpenGL中，Z轴是向摄像机的反方向，所以我们需要翻转Z轴
    new_points[:, 2] *= -1
    new_points[:, :2] = new_points[:, [1, 0]]

    return new_points

def main(args):
    # datapaser_tranforms = json.load(open(join(args.input, 'dataparser_transforms.json')))
    # scale = datapaser_tranforms['scale']
    # global_trans = np.eye(4)
    # global_trans[:3, :4] = np.asarray(datapaser_tranforms['transform'])
    
    # transform2opencv = lambda x: x
    
    input_transforms = json.load(open(join(args.input, 'transforms.json')))
    input_poses = [transform2opencv(np.asarray(transform['transform_matrix'])) for transform in input_transforms['frames']]
    input_xyz = np.asarray([pose[:3, 3] for pose in input_poses])
    input_pcd = pcd_array2o3d(input_xyz)
    input_scene_pcd = o3d.io.read_point_cloud(join(args.input, 'exports/pcd/point_cloud.ply'))
    points = np.asarray(input_scene_pcd.points)
    points = transform_points_opengl_to_opencv(points)
    input_scene_pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([input_pcd, input_scene_pcd])
    # vis_point_cloud(input_xyz)
    
    tranforms = json.load(open(join(args.input, 'camera_paths', args.transform_name)))
    vis_poses = [transform2opencv(np.asarray(transform['camera_to_world']).reshape(4, 4)) for transform in tranforms['camera_path']]
    # vis_poses = [np.linalg.inv(np.linalg.inv(pose) @ np.linalg.inv(global_trans * scale)) for pose in vis_poses]
    # vis_scene_pcd = o3d.io.read_point_cloud(join(args.input, 'point_cloud.ply'))
    vis_xyz = np.asarray([pose[:3, 3] for pose in vis_poses])
    vis_pcd = pcd_array2o3d(vis_xyz)
    
    
    
    o3d.visualization.draw_geometries([vis_pcd, input_pcd, input_scene_pcd])

if __name__ == '__main__':
    args = parse_args()
    main(args)