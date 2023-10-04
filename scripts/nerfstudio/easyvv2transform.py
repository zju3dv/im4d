import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from lib.utils.data_utils import read_camera
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def main(args):
    extri_path = join(args.input, 'extri.yml')
    intri_path = join(args.input, 'intri.yml')
    cams = read_camera(intri_path, extri_path)
    
    # transform.json
    frames = []
    for idx, cam in enumerate(cams.keys()):
        if cam == 'basenames':
            continue
        file_path = join('images', f'{cam}.jpg')
        colmap_im_id = idx
        
        w2c = cams[cam]['RT']
        
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1
        transform_matrix = c2w
        
        frame = {
            'file_path': file_path,
            'transform_matrix': transform_matrix.tolist(),
            'colmap_im_id': colmap_im_id,
            'camera_model': 'OPENCV',
            'fl_x': cams[cam]['K'][0, 0],
            'fl_y': cams[cam]['K'][1, 1],
            'cx': cams[cam]['K'][0, 2],
            'cy': cams[cam]['K'][1, 2],
            'w': 1920,
            'h': 1080,
            'k1': cams[cam]['dist'][0, 0],
            'k2': cams[cam]['dist'][0, 1],
            'p1': cams[cam]['dist'][0, 2],
            'p2': cams[cam]['dist'][0, 3],
            'k3': cams[cam]['dist'][0, 4],
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

