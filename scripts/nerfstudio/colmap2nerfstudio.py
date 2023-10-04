import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

# from lib.utils.data_utils import read_camera
from trdparty.colmap.read_write_model import read_model, qvec2rotmat
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--input_list', type=str, default=None)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def main(args):
    cameras, images, point3Ds = read_model(args.input)
    if args.input_list is not None:
        interested = open(args.input_list).read().splitlines()
    else:
        interested = []
    
    frames = []
    
    for k, v in tqdm(images.items()):
        if len(interested) != 0 and v.name not in interested:
            continue
        file_path = join('images', v.name)
        colmap_im_id = k
        
        w2c = np.eye(4)
        w2c[:3, :3] = v.qvec2rotmat()
        w2c[:3, 3] = v.tvec
        
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1
        transform_matrix = c2w
        
        cam_id = v.camera_id
        
        frame = {
            'file_path': file_path,
            'transform_matrix': transform_matrix.tolist(),
            'colmap_im_id': colmap_im_id,
            'camera_model': 'OPENCV',
            'fl_x': cameras[v.camera_id].params[0],
            'fl_y': cameras[v.camera_id].params[1],
            'cx': cameras[v.camera_id].params[2],
            'cy': cameras[v.camera_id].params[3],
            'w': cameras[cam_id].width,
            'h': cameras[cam_id].height,
            'k1': cameras[cam_id].params[4],
            'k2': cameras[cam_id].params[5],
            'p1': cameras[cam_id].params[6],
            'p2': cameras[cam_id].params[7],
            'k3': 0,
        }
        frames.append(frame)
    out = {}
    out['frames'] = frames
    
    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1
    out["applied_transform"] = applied_transform.tolist()
    with open(join(f'{args.output}', 'transforms.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4)
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
