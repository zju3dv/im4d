import numpy as np
import cv2
import random
import json
from torch import nn
import torch
from os.path import join
import trimesh
# from imgaug import augmenters as iaa
# from lib.config import cfg
from plyfile import PlyData
# from lib.utils import data_config
import re
import os
from io import BytesIO
from typing import Union, List, Tuple
import imageio
import torch.nn.functional as F
from lib.utils.parallel_utils import async_call
# from lib.utils.nerfstudio import colormaps

def read_bbox_from_dir(dir_path, frames, format='{:05d}.ply', padding_size=0.1, frame_start_number=1):
    if os.path.exists(join(dir_path, 'bbox_{}_{}_{}.txt'.format(padding_size, frames[0], frames[1]))):
        print('Load bbox from {}'.format(join(dir_path, 'bbox_{}_{}_{}.txt'.format(padding_size, frames[0], frames[1]))))
        return np.loadtxt(join(dir_path, 'bbox_{}_{}_{}.txt'.format(padding_size, frames[0], frames[1]))).astype(np.float32)
    import trimesh 
    bounds = []
    for frame_id in range(frames[0], frames[1]):
        if format[-3:] == 'npy':
            points = np.asarray(np.load(join(dir_path, format.format(frame_id + frame_start_number))))
        else:
            points = np.asarray(trimesh.load(join(dir_path, format.format(frame_id + frame_start_number))).vertices)
        # transform = np.array(cfg.transform)
        # points = points @ transform[:3, :3].T + transform[:3, 3:].T
        bound = np.array([points.min(0)-padding_size, points.max(0)+padding_size]).astype(np.float32)
        bounds.append(bound)
    bound = np.array([np.array(bounds).min(axis=(0, 1)), np.array(bounds).max(axis=(0, 1))]).astype(np.float32)
    try:
        print('Save bbox to {}'.format(join(dir_path, 'bbox_{}_{}_{}.txt'.format(padding_size, frames[0], frames[1]))))
        np.savetxt(join(dir_path, 'bbox_{}_{}_{}.txt'.format(padding_size, frames[0], frames[1])), bound)
        return np.loadtxt(join(dir_path, 'bbox_{}_{}_{}.txt'.format(padding_size, frames[0], frames[1]))).astype(np.float32)
    except:
        return bound
        
def export_dir_to_video(img_dir, video_path, fps=30, format='.png', quality=8.5, crf=8):
    import shutil
    if shutil.which('ffmpeg') is not None:
        cmd = 'ffmpeg -loglevel error -y -framerate {} -f image2 -pattern_type glob -nostdin -y -r {} -i "{}/*{}" -c:v libx264 -crf {} -pix_fmt yuv420p {}'.format(fps, fps, img_dir, format, crf, video_path)
        # cmd = 'ffmpeg -loglevel error -y -framerate {} -f image2 -pattern_type glob -nostdin -y -r {} -i "{}/*{}" -c:v libx264 -crf {} -pix_fmt yuv444p {}'.format(fps, fps, img_dir, format, crf, video_path)
        os.system(cmd)
    else:
        img_paths = [item.path for item in os.scandir(img_dir) if item.name.endswith(format)]
        img_paths = sorted(img_paths)
        imgs = []
        for img_path in img_paths:
            img = imageio.imread(img_path)
            if (img.shape) == 3:
                img = img[..., :3]
            else:
                img = img
            imgs.append(img)
        imageio.mimwrite(video_path, imgs, fps=fps, quality=quality)
            

def rgb_convert_to_img(img, acc=None):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255.).astype(np.uint8)
    if acc is not None:
        acc = (acc * 255.).astype(np.uint8)
        img = np.concatenate([img, acc[..., None]], axis=-1)
    return img

def dpt_convert_to_img(dpt, acc=None):
    min_val, max_val = dpt.min(), dpt.max()
    if acc is not None and (acc > 0.9).sum() > 0:
        min_val = dpt[acc>0.9].min()
    dpt = (dpt - min_val) / (max_val - min_val)
    dpt = np.clip(dpt, 0., 1.)
    dpt = dpt * 0.6 + 0.2
    
    dpt = cv2.applyColorMap((dpt*255.).astype(np.uint8), cv2.COLORMAP_JET)
    if acc is not None:
        dpt = np.concatenate([dpt, (acc * 255.).astype(np.uint8)[..., None]], axis=-1)
    return dpt


def load_img(img_path):
    return np.asarray(imageio.imread(img_path)).astype(np.uint8)

    
@async_call
def save_img(img_path, img, acc_map=None):
    if acc_map is not None:
        img = np.concatenate([img.astype(np.float32), acc_map.astype(np.float32)[..., None]], axis=-1)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255.).astype(np.uint8)
    os.system('mkdir -p {}'.format(os.path.dirname(img_path)))
    imageio.imwrite(img_path, img)
    
@async_call
def save_srcinps(img_path, src_inps):
    src_inp = np.concatenate([ src_inp.transpose(1, 2, 0) * 0.5 + 0.5 for src_inp in src_inps], axis=1)
    save_img(img_path, src_inp)
    
    
def gen_possible_depth(dpt, start=0, end=1.):
    dpt = dpt.copy()
    dpt = dpt * (end - start) + start
    return dpt

def gen_possible_depths(depth, acc_map):
    lengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    for start in np.linspace(0, 0.9, 10):
        for length in lengths:
            if start + length <= 1.:
                dpt = gen_possible_depth(depth, start, start + length)
                img = cv2.applyColorMap((dpt*255.).astype(np.uint8), cv2.COLORMAP_JET)
                print(start, start + length, dpt.min(), dpt.max())
                cv2.imwrite('depths/{}_{}.png'.format(start, start + length), np.concatenate([img, (acc_map[..., None] * 255.).astype(np.uint8)], axis=-1))
    
@async_call
def save_depth(img_path, img, acc_map=None, min_val=None, save_orig=False):
    if save_orig:
        np.savez_compressed(img_path.replace('.jpg', '.npz'), img=img, acc_map=acc_map)
    img = img.copy()
    if min_val is not None:
        img = np.clip(img, min_val, None)
    elif acc_map is not None:
        try:
            min_val = img[acc_map > 0.9].min()
        except:
            print('Warning: acc_map is all zero, use min_val=0.')
            min_val = 0.
        img = np.clip(img, min_val, None)
        # dpt = colormaps.apply_depth_colormap(torch.from_numpy(img)[..., None], torch.from_numpy(acc_map)[..., None]).numpy()
        # imageio.imwrite(img_path, (dpt * 255.).astype(np.uint8))
    img -= img.min()
    img /= (img.max() + 1e-6)
    img = img * 0.6 + 0.2 # 0.2 == 0.8
    # import ipdb; ipdb.set_trace()
    # gen_possible_depths(img, acc_map)
    # img = 1 - img
    # img = (img * 255.).astype(np.uint8)
    # save_img(img_path, img)
    # img0 = cv2.applyColorMap((img*255.).astype(np.uint8), cv2.COLORMAP_PLASMA)
    img = cv2.applyColorMap((img*255.).astype(np.uint8), cv2.COLORMAP_JET)
    # img0_path = img_path.replace('.jpg', '_plasma.jpg')
    # img_path = img_path
    if acc_map is not None:
        # img0 = np.concatenate([img0, (acc_map[..., None] * 255.).astype(np.uint8)], axis=-1)
        img = np.concatenate([img, (acc_map[..., None] * 255.).astype(np.uint8)], axis=-1)
        # img0_path = img0_path.replace('.jpg', '.png')
        img_path = img_path.replace('.jpg', '.png')
    # cv2.imwrite(img0_path, img0)
    os.system('mkdir -p {}'.format(os.path.dirname(img_path)))
    cv2.imwrite(img_path, img)
    # save_img(img_path, img)
    
def calib_color_img(img_ori, bgr_sol, transpose=True):
    rs = []
    for i in range(3):
        channel = np.array(img_ori[:,:,i],dtype=np.double)
        X = np.stack([channel**2,channel ,np.ones_like(channel)])
        if transpose:
            y = np.dot(bgr_sol[2-i].reshape(1,3),X.reshape(3,-1)).reshape(channel.shape)
        else:
            y = np.dot(bgr_sol[i].reshape(1,3),X.reshape(3,-1)).reshape(channel.shape)
        rs.append(y)
    rs_img = np.stack(rs,axis=2)
    rs_i = cv2.normalize(rs_img, None, alpha=rs_img.min(), beta=rs_img.max(), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return rs_i

def load_resize_undist_calib_im_bytes(imp: str, K: np.ndarray, D: np.ndarray, center_h_w: Union[int, List[int]], scale: Union[float, List[int]], ccm: np.ndarray,
                                encode_ext='.jpg', decode_flag=cv2.IMREAD_UNCHANGED, dist_opt_K: bool = False):
    # Load image -> resize -> undistort -> save to bytes (jpeg)
    img = load_image_from_bytes(load_image_bytes(imp), decode_flag=decode_flag)  # cv2 decoding (fast)
    img = calib_color_img(img, ccm)
    if np.linalg.norm(D) != 0.:
        img = cv2.undistort(img, K, D)
    oH, oW = img.shape[:2]
    if isinstance(scale, float): H, W = int(oH * scale), int(oW * scale)
    else: H, W = scale  # ratio is actually the target image size
    # rH, rW = H / oH, W / oW
    if oH != H or oW != W:
        if imp.split('.')[-1] == 'jpg':
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # H, W, 3, uint8
        else:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)  # H, W, 3, uint8
        
    if isinstance(center_h_w, int):
        tar_h, tar_w = center_h_w, center_h_w
    else:
        tar_h, tar_w = center_h_w
    if tar_h != -1:
        crop_h, crop_w = (H - tar_h)//2, (W - tar_w)//2
        crop_h_, crop_w_ = H - tar_h - crop_h, W - tar_w - crop_w
        img = img[crop_h:-crop_h_, crop_w:-crop_w_]
    _, buffer = cv2.imencode(encode_ext if imp.split('.')[-1] == 'jpg' else '.png', img)
    buffer: np.ndarray
    buffer = BytesIO(buffer)  # is this slow? tobytes and BytesIO
    return buffer


def load_resize_undist_schp_im_bytes(imp: str, K: np.ndarray, D: np.ndarray, center_h_w: Union[int, List[int]], scale: Union[float, List[int]],
                                encode_ext='.jpg', decode_flag=cv2.IMREAD_UNCHANGED, dist_opt_K: bool = False):
    # Load image -> resize -> undistort -> save to bytes (jpeg)
    img = load_image_from_bytes(load_image_bytes(imp), decode_flag=decode_flag)  # cv2 decoding (fast)
    # for schp, we need to firstly convert it into binary mask, and then undistort
    msk = img.sum(axis=-1) != 0
    img = (msk * 255).astype(np.uint8)
    if np.linalg.norm(D) != 0.:
        img = cv2.undistort(img, K, D)
    oH, oW = img.shape[:2]
    if isinstance(scale, float): H, W = int(oH * scale), int(oW * scale)
    else: H, W = scale  # ratio is actually the target image size
    # rH, rW = H / oH, W / oW
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # H, W, 3, uint8
    
        
    if isinstance(center_h_w, int):
        tar_h, tar_w = center_h_w, center_h_w
    else:
        tar_h, tar_w = center_h_w
    if tar_h != -1:
        crop_h, crop_w = (H - tar_h)//2, (W - tar_w)//2
        crop_h_, crop_w_ = H - tar_h - crop_h, W - tar_w - crop_w
        img = img[crop_h:-crop_h_, crop_w:-crop_w_]
    _, buffer = cv2.imencode(encode_ext, img)
    buffer: np.ndarray
    buffer = BytesIO(buffer)  # is this slow? tobytes and BytesIO
    return buffer

def load_resize_undist_im_bytes(imp: str, K: np.ndarray, D: np.ndarray, center_h_w: Union[int, List[int]], scale: Union[float, List[int]],
                                encode_ext='.jpg', decode_flag=cv2.IMREAD_UNCHANGED, dist_opt_K: bool = False):
    # Load image -> resize -> undistort -> save to bytes (jpeg)
    img = load_image_from_bytes(load_image_bytes(imp), decode_flag=decode_flag)  # cv2 decoding (fast)
    if np.linalg.norm(D) != 0.:
        img = cv2.undistort(img, K, D)
    oH, oW = img.shape[:2]
    if isinstance(scale, float): H, W = int(oH * scale), int(oW * scale)
    else: H, W = scale  # ratio is actually the target image size
    # rH, rW = H / oH, W / oW
    if imp.split('.')[-1] == 'jpg':
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # H, W, 3, uint8
    else:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)  # H, W, 3, uint8
    
        
    if isinstance(center_h_w, int):
        tar_h, tar_w = center_h_w, center_h_w
    else:
        tar_h, tar_w = center_h_w
    if tar_h != -1:
        crop_h, crop_w = (H - tar_h)//2, (W - tar_w)//2
        img = img[crop_h:crop_h+tar_h, crop_w:crop_w+tar_w]
    if encode_ext is not None:
        _, buffer = cv2.imencode(encode_ext if imp.split('.')[-1] == 'jpg' else '.png', img)
        buffer: np.ndarray
        buffer = BytesIO(buffer)  # is this slow? tobytes and BytesIO
        return buffer
    else:
        return img

def load_image_from_bytes(buffer: np.ndarray, decode_flag=cv2.IMREAD_UNCHANGED, debug=False):
    if isinstance(buffer, BytesIO):
        buffer = buffer.getvalue()  # slow? copy?
    if isinstance(buffer, bytes):
        image: np.ndarray = cv2.imdecode(np.frombuffer(buffer, np.uint8), decode_flag)  # MARK: 10-15ms
    else:
        image = buffer
    return image

def load_image_bytes(im: str):
    with open(im, "rb") as fh:
        buffer = BytesIO(fh.read())
    return buffer

def parse_cameras_genebody(cams):
    basenames, c2ws, ixts, Ds = [], [], [], []
    for k in cams:
        basenames.append(k)
        c2ws.append(cams[k]['c2w'])
        ixts.append(cams[k]['K'])
        Ds.append(cams[k]['D'])
    exts = np.linalg.inv(np.array(c2ws)).astype(np.float32)
    return basenames, \
           np.array(exts).astype(np.float32), \
           np.array(c2ws).astype(np.float32), \
           np.array(ixts).astype(np.float32), \
           np.array(Ds).astype(np.float32)

def add_batch(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [add_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch_ = {}
        for key in batch:
            batch_[key] = add_batch(batch[key])
        batch = batch_
    elif isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        batch = batch[None]
    else:
        batch = torch.tensor(batch)[None]
    return batch

def get_bound_2d_mask(bounds, K, H, W):
    points = bounds[:, :3] @ K.T
    corners_2d = points[..., :2] / points[..., 2:]
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_bound_2d_mask_padding(bounds, K, H, W, padding=0.1):
    bounds = bounds.coppy()
    points = bounds[:, :3] @ K.T
    corners_2d = points[..., :2] / points[..., 2:]
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def carve_vhull(
    H: torch.Tensor, W: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
    bounds: torch.Tensor, msk: torch.Tensor, voxel_size: float = 0.10, padding: float = 0.10,
    vhull_thresh: float = 0.75, count_thresh: int = 1,
):
    # Project these points onto the camera
    xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)[None]  # homo
    cam = xyz1 @ affine_padding(torch.cat([R, T], dim=-1)).mT  # N, P, 4 @ N, 3, 4 -> N, P, 4, homo camera coordinates
    cam = cam[..., :-1]  # last dim will always be one for valid affine transforms
    pix = cam @ K.mT  # N, P, 3 @ N, 3, 3, homo pixel coords
    pix = pix[..., :-1] / pix[..., -1:]  # N, P, 2, pixel coords

    # Use max size, instead of the one passed in before sampling
    pixel_uv_range = pix / torch.stack([W, H], dim=-1)[..., None, :] * 2 - 1  # N, P, 2 to sample the msk
    should_count_camera = ((pixel_uv_range > -1.0) & (pixel_uv_range < 1.0)).all(dim=-1)  # N, P
    vhull_camera_count = should_count_camera.sum(dim=0)  # P,

    H, W = msk.shape[-3:-1]  # sampling size (the largest of all images)
    # pix = pixel_uv_range
    pix = pix / msk.new_tensor([W, H]) * 2 - 1  # N, P, 2 to sample the msk (dimensionality normalization for sampling)
    valid = F.grid_sample(msks.permute(0, 3, 1, 2), pix[:, None], align_corners=True)[:, 0, 0]  # whether this is background
    valid = (valid > 0.5).float().sum(dim=0)  # N, 1, 1, P -> N, P
    valid = (valid / vhull_camera_count > vhull_thresh) & (vhull_camera_count > count_thresh)  # P, ratio of cameras sees this

    # Find valid points on the voxels
    import ipdb; ipdb.set_trace()
    inds = valid.nonzero()  # MARK: SYNC
    vhull = world_xyz.reshape(-1, 3)[inds]
    vhull = remove_outlier(vhull[None], K=5, std_ratio=5.0)[0]
    import ipdb; ipdb.set_trace()

    bounds = torch.stack([vhull.min(dim=0)[0], vhull.max(dim=0)[0]])
    bounds = bounds + bounds.new_tensor([-padding, padding])[:, None]

    return vhull, bounds

def get_padding_masks(msks):
    hs, ws = [msk.shape[1] for msk in msks], [msk.shape[2] for msk in msks]
    hmax, wmax = np.max(hs), np.max(ws)
    msk_len = len(msks)
    new_msks = torch.zeros((msk_len, hmax, wmax), dtype=torch.float32, device=msks[0].device)
    for i in range(msk_len):
        new_msks[i, :hs[i], :ws[i]] = msks[i][0]
    return new_msks

def multi_indexing(indices: torch.Tensor, shape: torch.Size, dim=-2):
    # index will first be augmented to match the values' dimentionality at the back
    # then we will try to broatcast index's shape to values shape
    shape = list(shape)
    back_pad = len(shape) - indices.ndim
    for _ in range(back_pad): indices = indices.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return indices.expand(*expand_shape)

def multi_gather(values: torch.Tensor, indices: torch.Tensor, dim=-2):
    # Gather the value at the -2th dim of values, augment index shape on the back
    # Example: values: B, P, 3, index: B, N, -> B, N, 3
    # index will first be augmented to match the values' dimentionality at the back
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the values's second to last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(indices, values.shape, dim))

def remove_outlier(pts: torch.Tensor, K: int = 20, std_ratio=2.0, ret_inds=False):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.view(-1, 3).detach().cpu().numpy())
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=K, std_ratio=std_ratio)
    if not ret_inds:
        return torch.as_tensor(np.array(pcd.points)[np.array(ind)]).to(pts.device, pts.dtype, non_blocking=True).view(pts.shape[0], -1, 3)
    else:
        return torch.as_tensor(np.array(pcd.points)[np.array(ind)]).to(pts.device, pts.dtype, non_blocking=True).view(pts.shape[0], -1, 3), ind
        

def get_grid_from_mask(batch, vhull_thresh=0.9, count_thresh=6):
    msks, exts, ixts = batch['msks'], batch['exts'], batch['ixts']
    hs, ws = torch.as_tensor([msk.shape[1] for msk in msks], device=exts.device), torch.as_tensor([msk.shape[2] for msk in msks], device=exts.device)
    msks = get_padding_masks(msks)[..., None]
    pts = batch['pts']
    world_xyz = pts[0, ..., 0, :] 
    xyz = world_xyz.reshape(-1, 3)
    xyz = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
    xyz = xyz[None] @ exts[0].mT 
    pix = xyz[:, :, :3] @ ixts[0].mT
    pix = pix[..., :-1] / pix[..., -1:]  # N, P, 2, pixel coords

    # Use max size, instead of the one passed in before sampling
    pixel_uv_range = pix / torch.stack([ws, hs], dim=-1)[..., None, :] * 2 - 1  # N, P, 2 to sample the msk
    should_count_camera = ((pixel_uv_range > -1.0) & (pixel_uv_range < 1.0)).all(dim=-1)  # N, P
    vhull_camera_count = should_count_camera.sum(dim=0)  # P,
    
    H, W = msks.shape[-3:-1]
    pix = pix / msks.new_tensor([W, H]) * 2 - 1  # N, P, 2 to sample the msk (dimensionality normalization for sampling)
    valid = F.grid_sample(msks.permute(0, 3, 1, 2), pix[:, None], align_corners=True)[:, 0, 0]  # whether this is background
    valid = (valid > 0.5).float().sum(dim=0)  # N, 1, 1, P -> N, P
    valid = (valid / vhull_camera_count > vhull_thresh) & (vhull_camera_count > count_thresh)  # P, ratio of cameras sees this
    
    inds = valid.nonzero()  # MARK: SYNC
    vhull = multi_gather(world_xyz.reshape(-1, 3), inds)  # P, 3; V -> V, 3
    vhull, valid_inds = remove_outlier(vhull[None], K=5, std_ratio=5.0, ret_inds=True)
    inds = inds[torch.tensor(valid_inds, device=inds.device, dtype=torch.long)]
    
    binary = torch.zeros_like(world_xyz[..., 0])
    binary.view(-1, 1)[inds] = 1.
    # import ipdb; ipdb.set_trace()
    if cfg.padding_grid:
        structuring_element = torch.ones(1, 1, 3, 3, 3, device=binary.device)
        dilated_tensor = F.conv3d(binary[None, None].float(), structuring_element.float(), padding=1) > 0 
        binary = dilated_tensor[0, 0].bool()
    return {'binary': binary.detach().cpu().numpy(), 'bounds': batch['bounds'][0].detach().cpu().numpy()}
    
    

def save_ply(points, ply_path, num_clusters=10):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters).fit(np.asarray(pcd.points))
    labels = kmeans.labels_
    colors = np.random.rand(num_clusters, 3)
    point_colors = colors[labels]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    if os.path.dirname(ply_path) != '':
        os.system('mkdir -p ' + os.path.dirname(ply_path))
    o3d.io.write_point_cloud(ply_path, pcd)

def read_and_denoise(ply_path, vox_size=0.02):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    # import ipdb; ipdb.set_trace()
    # o3d.visualization.draw_geometries([pcd])
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=vox_size)
    # o3d.visualization.draw_geometries([voxel_down_pcd])

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
    # display_inlier_outlier(voxel_down_pcd, ind)
    return voxel_down_pcd.select_by_index(ind)




def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    return intrinsics, extrinsics, depth_min

def read_pmn_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[1])
    return intrinsics, extrinsics, depth_min, depth_max

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def to_tensor(rgb):
    rgb = rgb.astype(np.float32) / 255.
    rgb = (rgb-data_config.mean_rgb) / data_config.std_rgb
    return rgb.transpose(2, 0, 1)

def to_img(tensor):
    tensor = tensor.detach().cpu().clone().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = tensor * data_config.std_rgb + data_config.mean_rgb
    tensor *= 255.
    return tensor.astype(np.uint8)


def resize_images(imgs, masks, ixt, input_size):
    ori_h, ori_w = imgs[0].shape[0], imgs[0].shape[1]
    tar_h, tar_w = input_size
    imgs = [cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR) for img in imgs]
    masks = [cv2.resize(mask.astype(np.uint8), input_size, interpolation=cv2.INTER_NEAREST) for mask in masks]
    ixt[0][0] *= (tar_h/ori_h)
    ixt[0][2] *= (tar_h/ori_h)
    ixt[1][1] *= (tar_w/ori_w)
    ixt[1][2] *= (tar_w/ori_w)
    return imgs, masks, ixt

def resize_image(img, mask, ixt, input_size):
    ori_h, ori_w = img.shape[0], img.shape[1]
    tar_h, tar_w = input_size
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), input_size, interpolation=cv2.INTER_NEAREST)
    ixt[0][0] *= (tar_h/ori_h)
    ixt[0][2] *= (tar_h/ori_h)
    ixt[1][1] *= (tar_w/ori_w)
    ixt[1][2] *= (tar_w/ori_w)
    return img, mask, ixt

def load_matrix(path):
    lines = [[float(w) for w in line.strip().split()] for line in open(path)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)

def load_nsvf_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        if intrinsics.shape[0] == 1 and intrinsics.shape[1] == 16:
            intrinsics = intrinsics.reshape(4, 4)
        return intrinsics
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, 'r') as file:

        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])
    return full_intrinsic

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        r3 = min(r1, r2)
    else:
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_distribution(heatmap, center, sigma_x, sigma_y, rho, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), (sigma_x/3, sigma_y/3), rho)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_heatmap_np(hm, point, box_size):
    """point: [x, y]"""
    # radius = gaussian_radius(box_size)
    radius = box_size[0]
    radius = max(0, int(radius))
    ct_int = np.array(point, dtype=np.int32)
    draw_umich_gaussian(hm, ct_int, radius)
    return hm


def get_edge(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return mask - cv2.erode(mask, kernel)


def compute_gaussian_1d(dmap, sigma=1):
    """dmap: each entry means a distance"""
    prob = np.exp(-dmap / (2 * sigma * sigma))
    prob[prob < np.finfo(prob.dtype).eps * prob.max()] = 0
    return prob


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


def homography_transform(pt, H):
    """pt: [n, 2]"""
    pt = np.concatenate([pt, np.ones([len(pt), 1])], axis=1)
    pt = np.dot(pt, H.T)
    pt = pt[..., :2] / pt[..., 2:]
    return pt


def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def blur_aug(inp):
    if np.random.random() < 0.1:
        if np.random.random() < 0.8:
            inp = iaa.blur_gaussian_(inp, abs(np.clip(np.random.normal(0, 1.5), -3, 3)))
        else:
            inp = iaa.MotionBlur((3, 15), (-45, 45))(images=[inp])[0]


def gaussian_blur(image, sigma):
    from scipy import ndimage
    if image.ndim == 2:
        image[:, :] = ndimage.gaussian_filter(image[:, :], sigma, mode="mirror")
    else:
        nb_channels = image.shape[2]
        for channel in range(nb_channels):
            image[:, :, channel] = ndimage.gaussian_filter(image[:, :, channel], sigma, mode="mirror")


def inter_from_mask(pred, gt):
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)
    intersection = np.logical_and(gt, pred).sum()
    return intersection


def draw_poly(mask, poly):
    cv2.fillPoly(mask, [poly], 255)
    return mask


def inter_from_poly(poly, gt, width, height):
    mask_small = np.zeros((1, height, width), dtype=np.uint8)
    mask_small = draw_poly(mask_small, poly)
    mask_gt = gt[..., 0]

    return inter_from_mask(mask_small, mask_gt)


def inter_from_polys(poly, w, h, gt_mask):
    inter = inter_from_poly(poly, gt_mask, w, h)
    if inter > 0:
        return False
    return True


def select_point(shape, poly, gt_mask):
    for i in range(cfg.max_iter):
        y = np.random.randint(shape[0] - poly['bbox'][3])
        x = np.random.randint(shape[1] - poly['bbox'][2])
        delta = np.array([poly['bbox'][0] - x, poly['bbox'][1] - y])
        poly_move = np.array(poly['poly']) - delta
        inter = inter_from_polys(poly_move, shape[1], shape[0], gt_mask)
        if inter:
            return x, y
    x, y = -1, -1
    return x, y


def transform_small_gt(poly, box, x, y):
    delta = np.array([poly['bbox'][0] - x, poly['bbox'][1] - y])
    poly['poly'] -= delta
    box[:2] -= delta
    box[2:] -= delta
    return poly, box


def get_mask_img(img, poly):
    mask = np.zeros(img.shape[:2])[..., np.newaxis]
    cv2.fillPoly(mask, [np.round(poly['poly']).astype(int)], 1)
    poly_img = img * mask
    mask = mask[..., 0]
    return poly_img, mask


def add_small_obj(img, gt_mask, poly, box, polys_gt):
    poly_img, mask = get_mask_img(img, poly)
    x, y = select_point(img.shape, poly.copy(), gt_mask)
    if x == -1:
        box = []
        return img, poly, box
    poly, box = transform_small_gt(poly, box, x, y)
    _, mask_ori = get_mask_img(img, poly)
    gt_mask += mask_ori[..., np.newaxis]
    img[mask_ori == 1] = poly_img[mask == 1]
    return img, poly, box[np.newaxis, :], gt_mask


def get_gt_mask(img, poly):
    mask = np.zeros(img.shape[:2])[..., np.newaxis]
    for i in range(len(poly)):
        for j in range(len(poly[i])):
            cv2.fillPoly(mask, [np.round(poly[i][j]['poly']).astype(int)], 1)
    return mask


def small_aug(img, poly, box, label, num):
    N = len(poly)
    gt_mask = get_gt_mask(img, poly)
    for i in range(N):
        if len(poly[i]) > 1:
            continue
        if poly[i][0]['area'] < 32*32:
            for k in range(num):
                img, poly_s, box_s, gt_mask = add_small_obj(img, gt_mask, poly[i][0].copy(), box[i].copy(), poly)
                if len(box_s) == 0:
                    continue
                poly.append([poly_s])
                box = np.concatenate((box, box_s))
                label.append(label[i])
    return img, poly, box, label


def truncated_normal(mean, sigma, low, high, data_rng=None):
    if data_rng is None:
        data_rng = np.random.RandomState()
    value = data_rng.normal(mean, sigma)
    return np.clip(value, low, high)


def _nms(heat, kernel=3):
    """heat: [b, c, h, w]"""
    pad = (kernel - 1) // 2

    # find the local minimum of heat within the neighborhood kernel x kernel
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def clip_to_image(bbox, h, w):
    bbox[..., :2] = torch.clamp(bbox[..., :2], min=0)
    bbox[..., 2] = torch.clamp(bbox[..., 2], max=w-1)
    bbox[..., 3] = torch.clamp(bbox[..., 3], max=h-1)
    return bbox


def load_ply(path):
    ply = PlyData.read(path)
    data = ply.elements[0].data
    x, y, z = data['x'], data['y'], data['z']
    model = np.stack([x, y, z], axis=-1)
    return model

def to_cuda(batch, device=torch.device('cuda:0')):
    if isinstance(batch, tuple) or isinstance(batch, list):
        #batch[k] = [b.cuda() for b in batch[k]]
        #batch[k] = [b.to(self.device) for b in batch[k]]
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict):
        #batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    else:
        # batch[k] = batch[k].cuda()
        batch = batch.to(device)
    return batch

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def get_ray_near_far_torch(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box in pytorch"""
    norm_d = torch.norm(ray_d, dim=-1, keepdim=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = torch.min(tmin, tmax)
    t2 = torch.max(tmin, tmax)
    near = torch.max(t1, dim=-1).values
    far = torch.min(t2, dim=-1).values
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box

def get_ray_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            if self.major_version == 4: # 4.4
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else: # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def get_mask(schp, mask):
    msk = schp.astype(np.int32)
    msk = (msk * [-1, 10, 100]).sum(axis=-1)
    palette = np.array([0, 128, 1280, 1408, 12800, 12928, 14080, 14208, 64,
                        192, 1344, 1472, 12864, 12992, 14144, 14272, 640, 768,
                        1920, 2048])
    leg_msk = (msk == palette[9]) | (msk == palette[16]) | (msk == palette[18])
    msk = (mask > 100).astype(np.uint8)
    msk[leg_msk] = 1
    return msk


def save_mesh_with_extracted_fields(ply_path, u, bounds, grid_resolution, threshold=0.5):
    import trimesh
    import mcubes
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    vertices = vertices / (np.array(grid_resolution)) * (bounds[1] - bounds[0])[None, :] + bounds[0][None, :]
    mesh = trimesh.Trimesh(vertices, triangles)
    os.system('mkdir -p {}'.format(os.path.dirname(ply_path)))
    mesh.export(ply_path)
    print('Save mesh to: {}'.format(ply_path))
    
    
def export_path_inp_cameras(input_cam, render_cam, path='test.ply'):
    print('Save render cams and input cams to: {}'.format(path))
    save_ply(np.concatenate([input_cam, render_cam], axis=0), path)
    
def get_space_points_torch(bounds, grid_resolution):
    x = torch.linspace(bounds[0, 0], bounds[1, 0], grid_resolution[0], device=bounds.device)
    y = torch.linspace(bounds[0, 1], bounds[1, 1], grid_resolution[1], device=bounds.device)
    z = torch.linspace(bounds[0, 2], bounds[1, 2], grid_resolution[2], device=bounds.device)
    X, Y, Z = torch.meshgrid(x, y, z)
    space_points = torch.stack([X, Y, Z], dim=-1)
    return space_points


def get_ratios(length, start_ratio, end_ratio, inter='linear'):
    funcs = {
        'linear': lambda x:x,
        'smoothstep': lambda x: x * x * (3 - 2 * x),
        'smootherstep': lambda x: x * x * x * (x * (x * 6 - 15) + 10),
        'smoothererstep': lambda x: x * x * x * x * (x * (x * (x * (-20) + 70) - 84) + 35)
    }
    weights = funcs[inter](np.linspace(0, 1, length))
    ratios = start_ratio + (end_ratio - start_ratio) * weights
    return ratios


def opencv2tranform(c2w):
    c2w = c2w.copy()
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    return c2w

def transform2opencv(c2w):
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    return c2w
    
def transform_points_opengl_to_opencv(points):
    new_points = points.copy()
    new_points[:, 2] *= -1
    new_points[:, :2] = new_points[:, [1, 0]]
    return new_points

def get_near_far(points, c2w, ixt, h, w):
    ext = np.linalg.inv(c2w)
    points = points @ ext[:3, :3].T + ext[:3, 3].T
    points = points @ ixt.T
    uv = points[:, :2] / points[:, 2:]
    uv_msk = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv_msk = np.logical_and(uv_msk, points[:, 2] > 0)
    assert(uv_msk.sum() > 0)
    z_vals = points[..., 2][uv_msk]
    near_far = np.asarray([np.percentile(z_vals, 0.1), np.percentile(z_vals, 99.9)])
    return near_far

def cv2gl(c2w):
    c2w = c2w.copy()
    c2w[0:3, 1:3] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[2, :] *= -1
    return c2w

def gl2cv(c2w):
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    return c2w

def load_path_pose(path, point_path):
    camera_path = json.load(open(path))
    h, w = camera_path['render_height'], camera_path['render_width']
    
    points = np.asarray(trimesh.load(point_path).vertices)
    points = transform_points_opengl_to_opencv(points)
    
    poses, ixts, near_fars = [], [], []
    for frame in camera_path['camera_path']:
        c2w = np.asarray(frame['camera_to_world']).reshape((4, 4))
        c2w = gl2cv(c2w)
        
        fov = frame['fov']
        # fx = 0.5 * w / np.tan(0.5 * fov / 180.0 * np.pi)
        fy = 0.5 * h / np.tan(0.5 * fov / 180.0 * np.pi)
        # aspect = frame['aspect']
        # 2 if aspect > 1.0:
            # fx = fy * aspect
        # fx = fx / aspect
        fx = fy
        # fy = fx
        # fx = fy
        ixt = np.asarray([fx, 0, w/2, 0, fy, h/2, 0, 0, 1]).reshape(3, 3)
        
        near_far = get_near_far(points, c2w, ixt, h, w)
        poses.append(np.linalg.inv(c2w).astype(np.float32))
        ixts.append(ixt.astype(np.float32))
        near_fars.append(near_far.astype(np.float32))
    return np.asarray(poses), np.asarray(ixts), np.asarray(near_fars), h, w


def read_bbox_renbody(data_root, frame_id, transform, bound_padding=0.1):
    pcd_path = join(data_root, 'pointcloud_denoise', '{:05d}.ply'.format(frame_id + 1))
    if os.path.exists(pcd_path): points = np.asarray(trimesh.load(pcd_path).vertices)
    else:
        ply_path = join(data_root, 'pointcloud', '{:05d}.ply'.format(frame_id + 1))
        points = np.asarray(read_and_denoise(ply_path).points)
        save_ply(points, ply_path.replace('pointcloud', 'pointcloud_denoise'))
    points = points @ transform[:3, :3].T + transform[:3, 3:].T
    bounds = np.array([points.min(0)-bound_padding, points.max(0)+bound_padding]).astype(np.float32)
    return bounds

def read_bbox_nhr(data_root, frame_id, transform, bound_padding=0.1):
    if 'NHR' in data_root: 
        if 'basketball' in data_root: pcd_path = join(data_root, 'vertices', '{:06d}.npy'.format(frame_id))
        else: pcd_path = join(data_root, 'vertices', '{}.npy'.format(frame_id))
    elif 'zju-mocap' in data_root: pcd_path = join(data_root, 'vertices', '{:06d}.npy'.format(frame_id))
    else: pass
    assert(os.path.exists(pcd_path))
    points = np.load(pcd_path)
    points = points @ transform[:3, :3].T + transform[:3, 3:].T
    bounds = np.array([points.min(0)-bound_padding, points.max(0)+bound_padding]).astype(np.float32)
    return bounds

def export_depth_to_pcd(depth, ixt, path='debug_depth.ply', crop_ratio=0.1):
    h, w = depth.shape
    crop_h, crop_w = int(crop_ratio * h), int(crop_ratio * w)
    ixt[0, 2] -= crop_w
    ixt[1, 2] -= crop_h
    tar_h, tar_w = int(h - crop_h*2), int(w - crop_w*2)
    depth = depth[crop_h:crop_h+tar_h, crop_w:crop_w+tar_w]
    h, w = depth.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = depth.reshape(-1)
    points = np.stack([X, Y, Z], axis=-1)
    points[..., :2] *= points[..., 2:]
    points = points @ np.linalg.inv(ixt).T
    save_ply(points, path)