'''
On the NHR dataset, mlp_maps found that the mask provided was relatively poor, so the edge part of the mask was excluded when testing PSNR.
'''

import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio as psnr_func
sys.path.append('.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    return args


def eval_img(pred_path, gt_path, msk_path, edge_path):
    pred = (imageio.imread(pred_path)[..., :3] / 255.).astype(np.float32)
    gt = (imageio.imread(gt_path) / 255.).astype(np.float32)
    msk = (imageio.imread(msk_path) / 255.)
    edge = (imageio.imread(edge_path) / 255.)
    
    eval_msk = np.logical_and(msk > 0.5, edge < 0.5)
    return psnr_func(pred[eval_msk], gt[eval_msk])
    # plt.imshow(eval_msk)
    # plt.savefig('test.jpg')
    
def main(args):
    psnrs = []
    for img_name in os.listdir(args.output_path):
        frame_id, view_id = int(img_name[5:9]), int(img_name[14:18])
        pred_path = join(args.output_path, img_name)
        gt_path = join(args.gt_path, '{:04d}_{:02d}_gt.png'.format(frame_id, view_id))
        msk_path = join(args.gt_path, '{:04d}_{:02d}_mask.png'.format(frame_id, view_id))
        edge_path = join(args.gt_path, '{:04d}_{:02d}_edge.png'.format(frame_id, view_id))
        psnrs.append(eval_img(pred_path, gt_path, msk_path, edge_path))
    print('Mean psnr: ', np.mean(psnrs))

if __name__ == '__main__':
    args = parse_args()
    main(args)