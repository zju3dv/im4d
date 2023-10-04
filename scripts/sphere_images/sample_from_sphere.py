import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

import torch
import torch.nn.functional as F
import cv2
import imageio
import glob
from lib.utils.parallel_utils import parallel_execution

np.random.seed(0)

def get_rays_d(h, w, ixt, R):
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z = np.ones_like(x)
    d = np.stack([x, y, z], axis=2)
    d = np.reshape(d, (-1, 3))
    d = d @ np.linalg.inv(ixt.T)
    d = d @ np.linalg.inv(R.T)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return d


def sample_envmap_image(image: torch.Tensor, ray_d: torch.Tensor):
    sh = ray_d.shape
    if image.ndim == 4:
        image = image[0]
    ray_d = ray_d.view(-1, 3)
    # envmap: H, W, C
    # viewdirs: N, 3

    # https://github.com/zju3dv/InvRender/blob/45e6cdc5e3c9f092b5d10e2904bbf3302152bb2f/code/model/sg_render.py
    image = image.permute(2, 0, 1).unsqueeze(0)

    theta = torch.arccos(ray_d[:, 2]).reshape(-1) - 1e-6
    phi = torch.atan2(ray_d[:, 1], ray_d[:, 0]).reshape(-1)  # 0 - pi

    # normalize to [-1, 1]
    query_y = (theta / torch.pi) * 2 - 1
    query_x = - phi / torch.pi
    grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0)

    rgb = F.grid_sample(image, grid, align_corners=False, padding_mode='border')
    rgb = rgb.squeeze().permute(1, 0)
    return rgb.view(sh)

def generate_rotation_matrices(num_samples):
    """
    生成旋转矩阵
    :param num_samples: 生成的视角数量
    :return: 旋转矩阵列表
    """
    # 生成球面均匀采样点
    theta = np.arccos(1 - 2 * np.random.rand(num_samples))  # 极角
    phi = 2 * np.pi * np.random.rand(num_samples)  # 方位角

    # 构建旋转矩阵
    rotation_matrices = []
    for i in range(num_samples):
        x = np.sin(theta[i]) * np.cos(phi[i])
        y = np.sin(theta[i]) * np.sin(phi[i])
        z = np.cos(theta[i])
        R = np.array([[np.cos(phi[i]), -np.sin(phi[i]), 0],
                      [np.sin(phi[i]), np.cos(phi[i]), 0],
                      [0, 0, 1]])
        V = np.array([[np.cos(theta[i]), 0, np.sin(theta[i])],
                      [0, 1, 0],
                      [-np.sin(theta[i]), 0, np.cos(theta[i])]])
        rotation_matrix = np.dot(R, V)
        rotation_matrices.append(rotation_matrix)

    return rotation_matrices

def generate_spherical_coordinates(num_samples):
    u = np.arange(0, num_samples+1) / num_samples
    v = np.arange(0, num_samples+1) / num_samples
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()
    # 将随机数映射到 [-1, 1] 范围内
    azimuth = 2 * np.pi * u
    elevation = np.arccos(2 * v - 1)

    # 将弧度转换为度
    a = np.degrees(azimuth)
    e = np.degrees(elevation)
    return a, e

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Street view image dir')
    parser.add_argument('--output', type=str, help='Output dir')
    parser.add_argument('--size', type=int, default=640, help='image size')
    parser.add_argument('--focal', type=float, default=256, help='focal length')
    args = parser.parse_args()
    return args


def generate_a_e():
    gap_a = 30
    gap_e = 45
    a = np.arange(60, 121, gap_a).astype(np.float32)
    e = np.arange(0, 360, gap_e).astype(np.float32)
    a, e = np.meshgrid(a, e)
    for i in range(a.shape[0]):
        a[i, :] += (np.random.random(len(a[i, :])) * 2 - 1) * gap_a / 2
    for i in range(e.shape[1]):
        e[:, i] += np.random.random(len(e[:, i])) * gap_e
    a, e = a.flatten(), e.flatten()
    return a, e

def process_single_image(img_path, output, h, w, ixt):
    output_dir = join(output, os.path.basename(img_path).split('.')[0])
    os.system(f'mkdir -p {output_dir}')
    a, e = generate_a_e()
    img = (np.array(imageio.imread(img_path)) / 255.).astype(np.float32)
    env_map = torch.tensor(img).float().cuda() if torch.cuda.is_available() else torch.tensor(img).float()
    for i,(a,e) in enumerate(zip(a, e)):
        a_rad = np.radians(a)
        e_rad = np.radians(e)
        y = a_rad
        pitch = np.pi / 2 - e_rad
        roll = 0.
        rotation_matrix_x = cv2.Rodrigues(np.array([a_rad, 0, 0]))[0]
        rotation_matrix_y = cv2.Rodrigues(np.array([0, pitch, 0]))[0]
        rotation_matrix_z = cv2.Rodrigues(np.array([0, 0, roll]))[0]
        rotation_matrix = rotation_matrix_z @ rotation_matrix_y @ rotation_matrix_x
        rays_d = torch.tensor(get_rays_d(h, w, ixt, rotation_matrix)).float().cuda() if torch.cuda.is_available() else torch.tensor(get_rays_d(h, w, ixt, rotation_matrix)).float()
        img = sample_envmap_image(env_map, rays_d)
        imageio.imwrite(f'{output_dir}/{i:03d}.jpg', (img.reshape(h,w,3).cpu().numpy() * 255.).astype(np.uint8))
        
def main(args):
    img_dir = args.input
    h, w = args.size, args.size
    focal = args.focal
    ixt = np.array([[focal, 0., w/2], [0, focal, h/2], [0, 0, 1 ]])
    
    img_paths = glob.glob(f'{img_dir}/*.jpg')
    img_paths = sorted(img_paths)
    # process_single_image(img_paths[0], args.output, h, w, ixt)
    # return
    parallel_execution(
        img_paths, 
        args.output,
        h, w, ixt,
        action=process_single_image,
        print_progress=True,
        num_processes=4,
    )
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

