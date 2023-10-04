import os
from os.path import join
import argparse
import sys
import numpy as np
from tqdm import tqdm
sys.path.append('.')

from trdparty.colmap.read_write_model import read_model, read_points3d_binary
import trimesh
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_sparse', type=str)
    parser.add_argument('--input_ply', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--thresh', type=float, default=3)
    args = parser.parse_args()
    return args

def main(args):
    '''
    generate the list of interested images from a COLMAP sparse model and ply.
    specifically, select images that rely on the point cloud of interest to complete the registration.
    '''
    cameras, images, points3D = read_model(args.input_sparse)
    roi_ply = trimesh.load_mesh(args.input_ply)
    roi_points = np.array(roi_ply.vertices)
    points = np.array([points3D[k].xyz for k in points3D])
    points_id = np.array([points3D[k].id for k in points3D])
    
    roi_points_id = []
    chunk = 128
    roi_points_cuda = torch.tensor(roi_points).cuda().float()
    points_cuda = torch.tensor(points).cuda().float()
    for i in tqdm(range(0, roi_points.shape[0], chunk)):
        chunk_points = roi_points_cuda[i:i+chunk]
        distance = (chunk_points[:, None] - points_cuda[None]).norm(dim=-1)
        roi_points_id += points_id[distance.argmin(dim=-1).cpu().numpy()].tolist()
    roi_points_id = np.array(roi_points_id)
    
    save_images = []
    save_conds = []
    for im_id, image in tqdm(images.items()):
        im_points_ids = image.point3D_ids[image.point3D_ids != -1]
        save_conds.append(np.isin(im_points_ids, roi_points_id).mean())
        save_images.append(image.name)
    threshs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
    for thresh in threshs:
        save_list = np.array(save_images)[(np.array(save_conds) > thresh)].tolist()
        print('thresh: {}, number of images: {}'.format(thresh, len(save_list)))
        open(join(args.output_path, '{}_{}.txt'.format(os.path.basename(args.input_ply).split('.')[0], thresh)), 'w').write('\n'.join(save_list))

if __name__ == '__main__':
    args = parse_args()
    main(args)





# def main(input_sparse, input_ply, output_path, thresh):
#     '''
#     Generate the list of interested images from a COLMAP sparse model and ply.
#     Specifically, select images that rely on the point cloud of interest to complete the registration.
#     '''

#     cameras, images, points3D = read_model(input_sparse)
#     roi_ply = trimesh.load_mesh(input_ply)
#     roi_points = np.array(roi_ply.vertices)
#     points = np.array([points3D[k].xyz for k in points3D])
#     points_id = np.array([points3D[k].id for k in points3D])

#     # Generate points_id for the interested point cloud
#     roi_points_id = []
#     chunk = 128 # Reduce it if CUDA out of memory occurs
#     roi_points_cuda = torch.tensor(roi_points).cuda().float()
#     points_cuda = torch.tensor(points).cuda().float()
#     for i in tqdm.tqdm(range(0, roi_points.shape[0], chunk)):
#         chunk_points = roi_points_cuda[i:i+chunk]
#         distance = (chunk_points[:, None] - points_cuda[None]).norm(dim=-1)
#         roi_points_id += points_id[distance.argmin(dim=-1).cpu().numpy()].tolist()
#     roi_points_id = np.array(roi_points_id)

#     # Select interested images
#     save_images = []
#     save_conds = []
#     for im_id, image in tqdm.tqdm(images.items()):
#         im_points_ids = image.point3D_ids[image.point3D_ids!=-1]
#         save_conds.append(np.isin(im_points_ids, roi_points_id).mean())
#         save_images.append(image.name)
#     save_list = np.array(save_images)[(np.array(save_conds) > thresh)].tolist()
#     print('Interested: {}'.format(len(save_list)))
#     open(output_path, 'w').writelines([item + '\n' for item in save_list])