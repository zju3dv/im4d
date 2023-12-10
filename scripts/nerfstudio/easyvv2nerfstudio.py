import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from lib.utils.data_utils import read_camera
from lib.utils.easyvv.easy_utils import read_camera
import json
import imageio

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--frame_id', type=int)
    parser.add_argument('--use_msk', type=bool, default=False)
    parser.add_argument('--write_alpha', type=bool, default=False)
    parser.add_argument('--white_bkgd', type=bool, default=False)
    parser.add_argument('--copy_images', type=bool, default=True)
    parser.add_argument('--parse_transform', action='store_true', help="Enable parse transform.")
    args = parser.parse_args()
    return args

def main(args):
    cams = os.listdir(join(args.input, 'images'))
    if args.copy_images:
        os.system(f'mkdir -p {args.output}/images')
        for cam in tqdm(cams):
            frame_path = join(args.input, 'images', cam, f'{args.frame_id:06d}.jpg')
            if not args.use_msk:
                tar_path = join(args.output, 'images', f'{cam}.jpg')
                os.system(f'cp {frame_path} {tar_path}')
            else:
                tar_path = join(args.output, 'images', f'{cam}.png') if args.write_alpha else join(args.output, 'images', f'{cam}.jpg')
                img = imageio.imread(frame_path)
                msk = np.array(imageio.imread(join(args.input, 'masks', cam, f'{args.frame_id:06d}.jpg')))
                if args.write_alpha:
                    out_img = np.concatenate([img, msk[:, :, None]], -1)
                    imageio.imwrite(tar_path, out_img)
                elif not args.write_alpha and args.white_bkgd:
                    msk = (msk / 255).astype(np.float32)
                    img = (img / 255).astype(np.float32)
                    out_img = img * msk[..., None] + (1 - msk[..., None])
                    out_img = (out_img * 255).astype(np.uint8)
                    imageio.imwrite(tar_path, out_img)
                elif not args.write_alpha and not args.white_bkgd:
                    msk = (msk / 255).astype(np.float32)
                    img = (img / 255).astype(np.float32)
                    out_img = img * msk[..., None]
                    out_img = (out_img * 255).astype(np.uint8)
                    imageio.imwrite(tar_path, out_img)
    if not args.parse_transform:
        return
    extri_path = join(args.input, 'extri.yml')
    intri_path = join(args.input, 'intri.yml')
    cams_ = cams
    cams = read_camera(intri_path, extri_path)
    
    # transform.json
    frames = []
    for idx, cam in enumerate(cams.keys()):
        if cam == 'basenames':
            continue
        file_path = join('images', f'{cam}.jpg' if not args.write_alpha else f'{cam}.png')
        colmap_im_id = idx
        
        w2c = cams[cam]['RT']
        
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1
        transform_matrix = c2w
        
        if cam not in cams_:
            continue
        
        frame = {
            'file_path': file_path,
            'transform_matrix': transform_matrix.tolist(),
            'colmap_im_id': colmap_im_id,
            'camera_model': 'OPENCV',
            'fl_x': cams[cam]['K'][0, 0],
            'fl_y': cams[cam]['K'][1, 1],
            'cx': cams[cam]['K'][0, 2],
            'cy': cams[cam]['K'][1, 2],
            'w': 640 if cams[cam]['W'] == -1 else cams[cam]['W'],
            'h': 480 if cams[cam]['H'] == -1 else cams[cam]['H'],
            'k1': cams[cam]['D'].reshape(5)[0],
            'k2': cams[cam]['D'].reshape(5)[1],
            'p1': cams[cam]['D'].reshape(5)[2],
            'p2': cams[cam]['D'].reshape(5)[3],
            'k3': cams[cam]['D'].reshape(5)[4],
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

