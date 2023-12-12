import h5py
import numpy as np
import cv2

import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from lib.utils.camera_utils import write_camera

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--output_basename', type=str)
    parser.add_argument('--json_file', type=str, default='transforms.json')
    args = parser.parse_args()
    return args

def h5_to_dict(f):

	ret = {}

	for key in f.keys():
		if isinstance(f[key], h5py.Dataset):
			ret[key] = np.array(f[key])
		else:
			ret[key] = h5_to_dict(f[key])

	return ret

def img_calib(img_bgr, bgr_calib):
    rs = []
    for i in range(3):
        channel = np.array(img_bgr[:, :, i],dtype=np.double)
        X = np.stack([channel ** 2,channel ,np.ones_like(channel)])
        y = np.dot(bgr_calib[i].reshape(1, 3),X.reshape(3, -1)).reshape(channel.shape)
        rs.append(y)
    rs_img = np.stack(rs,axis=2)
    rs_i = cv2.normalize(rs_img, None, alpha=rs_img.min(), beta=rs_img.max(), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return rs_i

def main(args):
    input_path = args.input
    _, basename = os.path.split(input_path)

    main_file = os.path.join(input_path, basename + '.smc')
    annots_file = os.path.join(input_path, basename + '_annots.smc')


    with h5py.File(annots_file, 'r') as f:
        data_annots = h5_to_dict(f)
    print('load annots from ', annots_file, ' ...')


    # Camera
    camera = {}
    cameras = data_annots['Camera_Parameter']
    CCMs = {}
    for camera_id in tqdm(cameras, desc='parse camera'):
        K = np.asarray(cameras[camera_id]['K'])
        dist = np.asarray([cameras[camera_id]['D']])
        RT = np.linalg.inv(np.asarray(cameras[camera_id]['RT'])) # c2w to w2c
        CCM = np.asarray(cameras[camera_id]['Color_Calibration'])

        camera[camera_id] = {
            'K': K,
            'dist': dist,
            'R': RT[:3, :3],
            'T': RT[:3, 3:],
        }

        CCMs[camera_id] = CCM

    keys = sorted(list(camera.keys()))
    camera = {key: camera[key] for key in keys}
    write_camera(camera, args.output, args.output_basename)


    # Mask
    msk_dir = os.path.join(input_path, 'masks')
    os.makedirs(msk_dir, exist_ok=True)

    cameras = data_annots['Mask']
    for camera_id in tqdm(cameras, desc='load masks'):
        camera_id_pad = '{:02d}'.format(int(camera_id))
        os.makedirs(os.path.join(msk_dir, camera_id_pad), exist_ok=True)
        frames = cameras[camera_id]['mask']
        for frame_id in frames:
            msk = cv2.imdecode(np.frombuffer(frames[frame_id], np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(msk_dir, camera_id_pad, '{:06d}.jpg'.format(int(frame_id))), msk)


    # Image
    with h5py.File(main_file, 'r') as f:
        data_image = h5_to_dict(f)

    img_dir = os.path.join(input_path, 'images')
    os.makedirs(img_dir, exist_ok=True)

    camera5mp = data_image['Camera_5mp']
    for camera_id in tqdm(camera5mp, desc='load images from camera 5mp'):
        camera_id_pad = '{:02d}'.format(int(camera_id))
        os.makedirs(os.path.join(img_dir, camera_id_pad), exist_ok=True)
        frames = camera5mp[camera_id]['color']
        for frame_id in frames:
            img = cv2.imdecode(np.frombuffer(frames[frame_id], np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_calib(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), CCMs[camera_id_pad]), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(img_dir, camera_id_pad, '{:06d}.jpg'.format(int(frame_id))), img)

    camera12mp = data_image['Camera_12mp']
    for camera_id in tqdm(camera12mp, desc='load images from camera 12mp'):
        camera_id_pad = '{:02d}'.format(int(camera_id))
        os.makedirs(os.path.join(img_dir, '{:02d}'.format(int(camera_id))), exist_ok=True)
        frames = camera12mp[camera_id]['color']
        for frame_id in frames:
            img = cv2.imdecode(np.frombuffer(frames[frame_id], np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_calib(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), CCMs[camera_id_pad]), cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(img_dir, camera_id_pad, '{:06d}.jpg'.format(int(frame_id))), img)

if __name__ == '__main__':
    args = parse_args()
    main(args)