import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from trdparty.colmap.read_write_model import read_model, qvec2rotmat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args

def main(args,
         num_thresh = 3, # 每个点最少应该被3+1张图片看到
         ):
    input_path = args.input
    cameras, images, points3D = read_model(join(input_path, 'dense', 'sparse'))
    cam_dict = {}
    
    for im_k in tqdm(images):
        
        ext = np.eye(4)
        ext[:3, :3] = qvec2rotmat(images[im_k].qvec)
        ext[:3, 3:] = images[im_k].tvec.reshape(3, 1)
        
        cam_params = cameras[images[im_k].camera_id].params
        assert(len(cam_params) == 4) # only support PINHOLE camera
        ixt = np.eye(3)
        ixt[0, 0], ixt[1, 1] = cam_params[0], cam_params[1]
        ixt[0, 2], ixt[1, 2] = cam_params[2], cam_params[3]
        
        h, w = cameras[images[im_k].camera_id].height, cameras[images[im_k].camera_id].width
        
        points = [points3D[k].xyz for k in images[im_k].point3D_ids if k != -1 and points3D[k] and len(points3D[k].image_ids) > num_thresh]
        points = np.array(points)
        
        if len(points) <= 50:
            continue
        points = points @ ext[:3, :3].T + ext[:3, 3:].T
        near_far = np.array([points[:, 2].min(), points[:, 2].max()])
        
        print(im_k, near_far)
    
        cam = {
            'img_path': images[im_k].name,
            'ext': ext,
            'ixt': ixt, 
            'near_far': near_far,
            'h': h,
            'w': w,
        }
        cam_dict[im_k] = cam
    np.save(join(input_path, 'annots/cam_dict.npy'), cam_dict)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    

