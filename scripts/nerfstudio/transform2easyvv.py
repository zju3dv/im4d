import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from lib.utils.data_utils import read_camera
from lib.utils.camera_utils import write_camera
from lib.utils.data_utils import transform2opencv, opencv2tranform
import json
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--output_basename', type=str)
    parser.add_argument('--json_file', type=str, default='transform.json')
    args = parser.parse_args()
    return args

def main(args):
    transforms = json.load(open(join(args.input, args.json_file)))
    cameras_out = {}
    cameras_new = {}
    for frame in transforms['frames']:
        key = os.path.basename(frame['file_path']).split('.')[0]
        if frame['camera_model'] == 'OPENCV':
            K = np.array([[frame['fl_x'], 0, frame['cx'], 0, frame['fl_y'], frame['cy'], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([[frame['k1'], frame['k2'], frame['p1'], frame['p2'], frame['k3']]])
        else:
            import ipdb; ipdb.set_trace()
        cameras_out[key] = {'K': K, 'dist': dist, 'H': frame['h'], 'W': frame['w']}
        
        c2w = np.array(frame['transform_matrix']).reshape(4, 4)
        c2w = transform2opencv(c2w)
        ext = np.linalg.inv(c2w)
        
        cam = cameras_out[key].copy()
        t = ext[:3, 3:]
        R = ext[:3, :3]
        cam['R'] = R
        # cam['Rvec'] = cv2.Rodrigues(R)[0]
        cam['T'] = t
        # mapkey[val.name.split('.')[0]] = val.camera_id
        
        cameras_new[key] = cam
        # cameras_new[val.name.split('.')[0].split('/')[0]] = cam
    keys = sorted(list(cameras_new.keys()))
    cameras_new = {key:cameras_new[key] for key in keys}
    write_camera(cameras_new, args.output, args.output_basename)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)

