import os
import cv2
import argparse
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_dilation

import sys
sys.path.append('.')

from lib.utils.easyvv.easy_utils import read_camera
from lib.utils.data_utils import save_mesh_with_extracted_fields
from lib.utils.im4d.fastrend_utils import FastRendUtils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input directory where intri.yml and extri.yml are stored')
    parser.add_argument('--frame_sample', nargs=3, type=int, help='start end and step of frame sample')
    parser.add_argument('--threshold', type=float, default=0.8, help='thershold for visual hull')
    parser.add_argument('--grid_resolution', nargs=3, type=int, default=[128, 128, 128], help='3D grid resolution')
    parser.add_argument('--min_bound', nargs=3, type=float, default=[-1, -1, -1], help='min bound of the scene')
    parser.add_argument('--max_bound', nargs=3, type=float, default=[ 1,  1,  1], help='max bound of the scene')
    parser.add_argument('--msk_format', type=str, default='{:04d}.png', help='name format of the mask file')
    args = parser.parse_args()
    return args

def getPose(args):
    intri_path = os.path.join(args.input, 'intri.yml')
    extri_path = os.path.join(args.input, 'extri.yml')
    cams = read_camera(intri_path, extri_path)

    basenames = sorted([k for k in cams])
    cam_len = len(basenames)

    ixts = np.array([cams[cam_id]['K'] for cam_id in basenames]).reshape(cam_len, 3, 3).astype(np.float32)

    exts = np.array([cams[cam_id]['RT'] for cam_id in basenames]).reshape(cam_len, 3, 4).astype(np.float32)
    exts_ones = np.zeros_like(exts[:, :1, :])
    exts_ones[..., 3] = 1.
    exts = np.concatenate([exts, exts_ones], axis=1)

    input_views = [int(basename) for basename in basenames]

    return basenames, exts, ixts, input_views

def compute_visual_hull(args):
    
    basenames, exts, ixts, input_views = getPose(args)

    exts = torch.Tensor(exts).cuda()
    ixts = torch.Tensor(ixts).cuda()

    msk_format = args.msk_format

    frame_sample = args.frame_sample
    grid_resolution = np.array(args.grid_resolution).astype(np.int32)
    global_bound = np.array([args.min_bound, args.max_bound]).astype(np.float32)

    threshold = args.threshold

    bbox_dir = os.path.join(args.input, 'vhull')
    os.makedirs(bbox_dir, exist_ok=True)

    for frame_id in tqdm(range(*frame_sample), desc='compute foreground bbox'):
        wpts = FastRendUtils.prepare_wpts(torch.from_numpy(global_bound), grid_resolution, (1, 1, 1)).cuda()
        wpts = wpts.squeeze(3)
        cnt = np.zeros(wpts.shape[:-1])

        for view_id in input_views:
            msk_path = os.path.join(args.input, 'masks', basenames[view_id], msk_format.format(frame_id))
            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            msk = (msk / 255.).astype(np.float32)
            h, w = msk.shape[:2]


            uv = wpts @ exts[view_id][:3, :3].T + exts[view_id][:3, 3:].T
            uv = uv @ ixts[view_id].T
            uv = uv[..., :2] / uv[..., 2:]
            uv = torch.round(uv).cpu().numpy().astype(np.int32)
            uv[..., 0] = uv[..., 0].clip(0, w - 1)
            uv[..., 1] = uv[..., 1].clip(0, h - 1)

            cnt += msk[uv[..., 1], uv[..., 0]]

        binary = (cnt >= threshold * len(input_views)).astype(bool)

        x, y, z = np.where(binary)
        bound = np.array([[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]])
        bound = global_bound[0] + (bound / grid_resolution) * (global_bound[1] - global_bound[0])

        np.save(os.path.join(bbox_dir, '{:06d}.npy'.format(frame_id)), bound)

def main(args):
    compute_visual_hull(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)