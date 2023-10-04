import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

import imageio
import cv2
import matplotlib.pyplot as plt 
import json
from matplotlib import cm

def is_overlapping(box1, box2, patch_size):
    x1, y1 = box1
    x2, y2 = box2

    overlap_x = (x1 < (x2 + patch_size)) and (x2 < (x1 + patch_size))
    overlap_y = (y1 < (y2 + patch_size)) and (y2 < (y1 + patch_size))
    
    return overlap_x and overlap_y

def find_most_dissimilar_patches(img1, img2, patch_size=128, num_patches=4):
    if img1.shape != img2.shape:
        raise ValueError("The two images must have the same shape.")

    diff = np.abs(img1 - img2)

    height, width = img1.shape[:2]
    best_patches = []
    best_scores = []
    candidate_patches = []

    for i in tqdm(range(0, height - patch_size), desc='traverse the image grid'):
        for j in range(0, width - patch_size):
            patch = diff[i:i+patch_size, j:j+patch_size]
            score = np.mean(patch)
            candidate_patches.append(((i, j), score))

    candidate_patches.sort(key=lambda x: x[1], reverse=True)
    
    for (i, j), score in tqdm(candidate_patches, desc='traverse patches'):
        if len(best_patches) >= num_patches:
            break
        if all([not is_overlapping((i, j), best_i_j, patch_size) for best_i_j, _ in best_patches]):
            best_patches.append(((i, j), score))

    return best_patches

# # 读取图像
# img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# best_patches, best_scores = find_most_dissimilar_patches(img1, img2, patch_size=128, num_patches=4)

# print("Most dissimilar patches at:")
# for (i, j), score in zip(best_patches, best_scores):
#     print(f"Top-left corner: ({i}, {j}), Score: {score}")
#     cv2.rectangle(img1, (j, i), (j+128, i+128), (255, 0, 0), 2)
#     cv2.rectangle(img2, (j, i), (j+128, i+128), (255, 0, 0), 2)

# cv2.imshow('Image 1', img1)
# cv2.imshow('Image 2', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def vis_patches(img1, img2, patches, patch_size=128, save_path=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_float = [(r, g, b) for r, g, b, _ in [cm.colors.to_rgba(c) for c in colors]]
    for idx, ((i, j), score) in enumerate(patches):
        color = colors_float[idx % len(colors_float)]
        cv2.rectangle(img1, (j, i), (j+patch_size, i+patch_size), color, 2)
        cv2.rectangle(img2, (j, i), (j+patch_size, i+patch_size), color, 2)
        
    imageio.imwrite(save_path, np.concatenate([img1, img2], axis=1))
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/mnt/data2/home/linhaotong/Datasets/dynerf/cut_beef')
    parser.add_argument('--output', type=str, default='/mnt/data2/home/linhaotong/Datasets/dynerf/cut_beef/im4d')
    parser.add_argument('--test_view', type=str, default='00')
    parser.add_argument('--frame_len', type=int, default=60)
    parser.add_argument('--eval_interval', type=int, default=30)
    parser.add_argument('--eval_ratio', type=float, default=0.5)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--num_patches', type=int, default=4)
    args = parser.parse_args()
    return args

def main(args):
    imgs = []
    for i in tqdm(range(args.frame_len)):
        img_path = join(args.input, 'images', args.test_view, '{:04d}.jpg'.format(i))
        img = (imageio.imread(img_path) / 255.).astype(np.float32)
        imgs.append(img)
        
    os.makedirs(join(args.output, 'test', 'patches'), exist_ok=True)
    os.makedirs(join(args.output, 'test', 'vis'), exist_ok=True)
    
    for i in tqdm(range(0, args.frame_len, args.eval_interval)):
        img = cv2.resize(imgs[i], None, fx=args.eval_ratio, fy=args.eval_ratio, interpolation=cv2.INTER_AREA)
        avg_img = cv2.resize(np.mean(imgs[i:i+args.eval_interval], axis=0), None, fx=args.eval_ratio, fy=args.eval_ratio, interpolation=cv2.INTER_AREA)
        patches = find_most_dissimilar_patches(img, avg_img, patch_size=args.patch_size, num_patches=args.num_patches)
        vis_patches(img, avg_img, patches, patch_size=args.patch_size, save_path=join(args.output, 'test', 'vis', '{:03d}.png'.format(i)))
        save_dict = {'coords': [patch[0] for patch in patches], 'patch_size': args.patch_size}
        save_path = join(args.output, 'test', 'patches', '{:03d}.json'.format(i))
        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(save_dict, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)