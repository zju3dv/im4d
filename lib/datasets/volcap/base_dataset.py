import numpy as np
import os
from lib.utils.base_utils import get_sample
from lib.utils.data_utils import load_image_bytes, load_image_from_bytes, load_resize_undist_im_bytes, load_img, load_resize_undist_calib_im_bytes, get_ratios, transform_points_opengl_to_opencv, get_near_far
from lib.utils.parallel_utils import parallel_execution
from lib.config import cfg, logger
import imageio
import cv2
from tqdm import tqdm
from lib.config import cfg
from lib.utils import data_utils
from lib.utils import base_utils
from lib.utils.easyvv import easy_utils
import trimesh
from lib.utils.base_utils import get_bound_corners
from os.path import join
from termcolor import colored
from torch.utils.data.dataloader import default_collate
from lib.utils.data_utils import parse_cameras_genebody, save_img
from lib.utils.pixel_sampling.pixel_sampler import PixelSampler
from lib.utils import rend_utils
from lib.config.yacs import CfgNode as CN

timer = base_utils.perf_timer(sync_cuda=False)

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        # Prepare camera intrinsic and extrinsic parameters
        # Prepare the metadata for rendering
        # Prepare bounding box or point cloud for near-far determination
        # Precompute some things in advance
        self.cfg = CN(kwargs)
        self.prepare_metainfo()
        self.prepare_camera() # prepare camera intrinsics and extrinsics
        self.prepare_render_meta() # prepare render meta for the rendered images
        if self.preload_data: self.preload_data_func()
        self.load_bboxes() # load bboxes
        self.load_pointcloud()
        self.precompute() # precompute some things in advance, e.g., near_far etc.

    def prepare_metainfo(self):
        self.data_root = os.path.join(cfg.workspace, self.cfg.data_root, cfg.scene)
        self.split = self.cfg.split
        self.start_frame = self.cfg.frame_sample[0]
        self.frame_len = self.cfg.frame_sample[1] - self.cfg.frame_sample[0]
        self.crop_h_w = self.cfg.get('crop_h_w', [-1, -1])
        self.imgs_per_batch = self.cfg.get('imgs_per_batch', 1)
        self.shift_pixel = self.cfg.get('shift_pixel', False)
        self.preload_data = self.cfg.get('preload_data', False)
        self.have_msk = self.cfg.get('msk_dir', 'none') != 'none'
        self.ignore_dist_k3 = self.cfg.get('ignore_dist_k3', False)
        self.img_dir = self.cfg.get('img_dir', 'images')
        self.msk_dir = self.cfg.get('msk_dir', 'masks')
        self.img_frame_format = self.cfg.get('img_frame_format', '{:06d}.jpg')
        self.msk_frame_format = self.cfg.get('msk_frame_format', '{:06d}.png')
        self.scene_type = self.cfg.get('scene_type', 'bbox_scene')
        self.near_far = np.array(self.cfg.get('near_far', [0.1, 100.]))
        self.bbox_type = self.cfg.get('bbox_type', 'NONE')
        self.render_path = cfg.get('render_path', False)
        self.path_type = self.cfg.get('path_type', 'NERFSTUDIO')
        self.path_name = self.cfg.get('path_name', '2023-07-13_220007.json')

    def prepare_camera(self):
        intri_path = join(self.data_root, self.cfg.get('intri_file', 'intri.yml'))
        extri_path = join(self.data_root, self.cfg.get('extri_file', 'extri.yml'))
        logger.debug('Loading camera from: {}, {}'.format(intri_path, extri_path))
        cams = easy_utils.read_camera(intri_path, extri_path)
        self.basenames = sorted([k for k in cams])
        cam_len = len(self.basenames)
        self.ixts = np.array([cams[cam_id]['K'] for cam_id in self.basenames]).reshape(cam_len, 3, 3).astype(np.float32)
        self.orig_ixts = self.ixts.copy()
        exts = np.array([cams[cam_id]['RT'] for cam_id in self.basenames]).reshape(cam_len, 3, 4).astype(np.float32)
        exts_ones = np.zeros_like(exts[:, :1, :])
        exts_ones[..., 3] = 1.
        self.exts = np.concatenate([exts, exts_ones], axis=1)
        transform = np.array(self.cfg.get('transform', np.eye(4)))
        self.exts = (self.exts @ np.linalg.inv(transform)).astype(np.float32)  # align z to up
        self.exts_inv = np.linalg.inv(self.exts).astype(np.float32)
        self.Ds = np.array([cams[cam_id]['D'] for cam_id in self.basenames]).reshape(cam_len, 5).astype(np.float32)
        if self.ignore_dist_k3: self.Ds[..., 4] = 0.

        # process ixts according to crop and resize
        special_views = self.cfg.get('special_views', [])
        special_views = get_sample(special_views)
        self.rs = [self.cfg.get('special_resize_ratio') if i in special_views else self.cfg.get('resize_ratio') for i in range(len(self.ixts))]
        self.ixts[:, :2] *= np.array(self.rs)[:, None, None]
        if self.crop_h_w[0] != -1:
            frame_id = self.start_frame
            img_paths = [join(self.data_root, self.img_dir, basename, self.img_frame_format.format(frame_id)) for basename in self.basenames]
            imgs = parallel_execution(img_paths, action=load_img, num_processes=32,print_progress=True, sequential=False, async_return=False, desc='loading images for compute ixts')
            tar_h, tar_w = self.crop_h_w
            new_ixts = []
            for idx, (img, ixt) in enumerate(zip(imgs, self.ixts)):
                h, w = img.shape[:2]
                if self.rs[idx] != 1.: h, w = int(h*self.rs[idx]), int(w*self.rs[idx])
                crop_h, crop_w = (h - tar_h)//2, (w - tar_w)//2
                ixt[0, 2] -= crop_w
                ixt[1, 2] -= crop_h
                new_ixts.append(ixt)
            self.ixts = new_ixts
        self.ixts = np.array(self.ixts).astype(np.float32)
        self.ixts_inv = np.linalg.inv(self.ixts).astype(np.float32)

    def prepare_render_meta(self):
        # mem bank
        self.meta_imgs_path = []
        self.meta_msks_path = []
        self.meta_viewids = []
        # index mem bank: given frame_id, and view_id, return the index in meta_imgs_path
        self.meta_idx = {}
        # render_meta
        self.meta = []

        render_views = get_sample(self.cfg.get('render_view_sample'))
        input_views = get_sample(self.cfg.get('input_view_sample'))
        test_views = self.cfg.get('test_views', [-1])
        if -1 not in test_views:
            render_views = test_views
            input_views = [view for view in input_views if view not in render_views]
            if self.split == 'train': render_views = input_views
        logger.info('Dataset split: {}, render views: {}'.format(self.split, str(render_views)))
        self.get_metas(render_views, input_views)

    def load_pointcloud(self):
        if 'scene_pcd' not in self.cfg: self.point_cloud = None; return
        pcd_path = join(self.data_root, 'nerfstudio/exports/pcd/{}'.format(self.cfg.scene_pcd))
        if os.path.exists(pcd_path):
            pcd = np.asarray(trimesh.load(pcd_path).vertices)
        else:
            pcd_path = join(self.data_root, self.cfg.scene_pcd)
            pcd = np.asarray(trimesh.load(pcd_path).vertices)
        if 'nerfstudio' in pcd_path:
            logger.debug('Transform point cloud from opengl to opencv')
            pcd = transform_points_opengl_to_opencv(pcd)
        num_max_pcd = 10000
        if pcd.shape[0] > num_max_pcd:
            np.random.seed(0)
            rand_msk = np.random.random(pcd.shape[0])
            pcd = pcd[rand_msk < (num_max_pcd/pcd.shape[0])]
        self.point_cloud = pcd

    def precompute(self, **kwargs):
        MAX_IMAGESIZE = 5120
        H, W = 5120, 5120  # MAX_IMGAESIZE
        self.coords = np.stack(np.meshgrid(np.linspace(
            0, W-1, W), np.linspace(0, H-1, H)), -1).astype(np.int32)
        self.coords = np.concatenate(
            [self.coords, np.ones_like(self.coords[..., :1])], axis=-1)

    def load_bboxes(self):
        # priority:
        # 1. per_frame bbox
        # 2. dataset bbox
        # 3. global bbox
        # TODO: implement priority
        frames = self.cfg.frame_sample
        self.corners = {}
        self.bounds = {}

        transform = np.array(self.cfg.get('transform', np.eye(4)))
        for frame_id in tqdm(np.arange(*frames), desc='loading bboxs'):
            if self.bbox_type == 'RENBODY': bounds = data_utils.read_bbox_renbody(self.data_root, frame_id, transform, cfg.bound_padding)
            elif self.bbox_type == 'NHR': bounds = data_utils.read_bbox_nhr(self.data_root, frame_id, transform, cfg.bound_padding)
            elif self.bbox_type == 'GENERAL': import ipdb; ipdb.set_trace()
            else: bounds = np.asarray(cfg.bounds).astype(np.float32)
            self.bounds[frame_id] = bounds
            self.corners[frame_id] = np.array(get_bound_corners(self.bounds[frame_id])).astype(np.float32)

        self.bound = np.array([np.array([self.bounds[k] for k in self.bounds]).min(axis=(0, 1)),
                               np.array([self.bounds[k] for k in self.bounds]).max(axis=(0, 1))]).astype(np.float32)
        self.corner = get_bound_corners(self.bound).astype(np.float32)

    def preload_img(self, frame_id, view_id, **kwargs):
        img_path = join(self.data_root, self.img_dir, self.basenames[view_id], self.img_frame_format.format(frame_id))
        if img_path in self.meta_imgs_path:
            return
        if self.have_msk:
            msk_path = join(self.data_root, self.msk_dir, self.basenames[view_id], self.msk_frame_format.format(frame_id))
            self.meta_msks_path.append(msk_path)
        self.meta_imgs_path.append(img_path)
        self.meta_viewids.append(view_id)
        if frame_id not in self.meta_idx: self.meta_idx[frame_id] = {}
        self.meta_idx[frame_id][view_id] = len(self.meta_imgs_path) - 1

    def get_render_path_meta(self, input_views, **kwargs):
        path_w2cs = rend_utils.create_center_radius(
            np.array(cfg.render_center),
            radius=cfg.render_radius,
            up=cfg.render_axis,
            ranges=cfg.render_view_range + [cfg.num_circle],
            angle_x=cfg.render_angle)  # path 1
        bottom = np.array([[0, 0, 0, 1.]])[None].repeat(cfg.num_circle, 0)
        path_w2cs = np.concatenate([path_w2cs, bottom], axis=1)
        path_w2cs = path_w2cs.astype(np.float32)
        path_c2ws = np.linalg.inv(path_w2cs)
        # DEBUG:
        # np.save('cams.npy', {'input': self.exts_inv, 'render': self.path_c2ws})
        # self.render_ids, render_id = [], 0

        self.render_exts = []
        self.render_ixts = []
        self.render_ids = []
        render_id = 0 # index
        for frame_id in tqdm(range(*cfg.render_frames), desc='loading render meta'):
            tar_ext = path_w2cs[render_id % cfg.num_circle]
            self.preload_img(frame_id, 0)
            self.meta += [(frame_id, -1)]
            self.render_exts.append(tar_ext.astype(np.float32))
            self.render_ixts.append(self.ixts[0].astype(np.float32))
            self.render_ids.append(render_id)
            render_id += 1

            if frame_id in cfg.render_stop_frame:
                for i in range(cfg.render_stop_time[cfg.render_stop_frame.index(frame_id)]):
                    tar_ext = path_w2cs[render_id % cfg.num_circle]
                    self.meta += [(frame_id, -1)]
                    self.render_exts.append(tar_ext.astype(np.float32))
                    self.render_ixts.append(self.ixts[0].astype(np.float32))
                    self.render_ids.append(render_id)
                    render_id += 1

        if len(cfg.render_ixts_start) >= 1:
            ixts_ratios = np.ones_like(np.array(self.render_ids)).astype(np.float32)
            prev = 1.
            for idx in range(len(cfg.render_ixts_start)):
                start = cfg.render_ixts_start[idx]
                end = cfg.render_ixts_end[idx]
                if idx >= 1: ixts_ratios[cfg.render_ixts_end[idx-1]:start] = prev
                ixts_ratios[start:end] = get_ratios(end-start, prev, cfg.render_ixts_ratio[idx], cfg.render_smooth)
                prev = cfg.render_ixts_ratio[idx]
            self.render_ixts = np.array(self.render_ixts).astype(np.float32)
            self.render_ixts[:, :2, :2] *= ixts_ratios[:, None, None]
            
        self.render_h_w = cfg.render_h_w
        self.render_exts = np.array(self.render_exts).astype(np.float32)
        self.render_ixts = np.array(self.render_ixts).astype(np.float32)
        self.render_exts_inv = np.linalg.inv(self.render_exts).astype(np.float32)
        self.render_ixts_inv = np.linalg.inv(self.render_ixts).astype(np.float32)

        # render_meta = {'render_exts': np.array(self.render_exts),
        #                'render_ixts': np.array(self.render_ixts),
        #                'frame_ids': [meta[0] for meta in self.meta],
        #                'src_views': [meta[2] for meta in self.meta]}
        # os.system('mkdir -p {}'.format(cfg.result_dir))
        # print(cfg.result_dir)
        # np.save(join(cfg.result_dir, 'render_meta.npy'), render_meta)
        # cfg_meta = {
        #     'render_center': cfg.render_center,
        #     'render_radius': cfg.render_radius,
        #     'render_axis': cfg.render_axis,
        #     'render_view_range': cfg.render_view_range,
        #     'render_angle': cfg.render_angle,
        #     'num_circle': cfg.num_circle,
        #     'render_frames': cfg.render_frames,
        #     'render_stop_frame': cfg.render_stop_frame,
        #     'render_stop_time': cfg.render_stop_time,
        #     'render_ixts_start': cfg.render_ixts_start,
        #     'render_ixts_end': cfg.render_ixts_end,
        #     'render_ixts_ratio': cfg.render_ixts_ratio,
        #     'render_smooth': cfg.render_smooth}
        # import json
        # json.dump(cfg_meta, open(
        #     join(cfg.result_dir, 'cfg.json'), 'w'), indent=4)

    def get_metas(self, render_views, input_views):
        if self.render_path:
            self.get_render_path_meta(input_views)
            return
        for frame_id in tqdm(range(*self.cfg.frame_sample), desc='loading meta'):
            for view_id in render_views:
                self.preload_img(frame_id, view_id)
                self.meta += [(frame_id, view_id)]

    def load_cameras(self, **kwargs):
        intri_path = join(self.data_root, 'intri.yml')
        extri_path = join(self.data_root, 'extri.yml')
        cams = easy_utils.read_camera(intri_path, extri_path)
        self.basenames = sorted([k for k in cams])
        cam_len = len(self.basenames)
        self.ixts = np.array([cams[cam_id]['K'] for cam_id in self.basenames]).reshape(
            cam_len, 3, 3).astype(np.float32)
        self.orig_ixts = self.ixts.copy()
        exts = np.array([cams[cam_id]['RT'] for cam_id in self.basenames]).reshape(
            cam_len, 3, 4).astype(np.float32)
        exts_ones = np.zeros_like(exts[:, :1, :])
        exts_ones[..., 3] = 1.
        self.exts = np.concatenate([exts, exts_ones], axis=1)
        transform = cfg.get('transform', np.eye(4))
        self.exts = (self.exts @ np.linalg.inv(transform)
                     ).astype(np.float32)  # align z to up
        self.exts_inv = np.linalg.inv(self.exts).astype(np.float32)
        self.Ds = np.array([cams[cam_id]['D'] for cam_id in self.basenames]).reshape(
            cam_len, 5).astype(np.float32)
        if kwargs.get('ignore_dist_k3', False):
            self.Ds[..., 4] = 0

    def preload_data_func(self):
        # preloading
        self.meta_imgs_buffer = parallel_execution(self.meta_imgs_path,
                                                   [self.orig_ixts[view_id]
                                                       for view_id in self.meta_viewids],
                                                   [self.Ds[view_id]
                                                       for view_id in self.meta_viewids],
                                                   [self.crop_h_w for i in range(
                                                       len(self.meta_viewids))],
                                                   # [self.ccms[view_id] for view_id in (self.meta_viewids)],
                                                   [self.rs[view_id]
                                                       for view_id in (self.meta_viewids)],
                                                   '.png',
                                                   action=load_resize_undist_im_bytes, num_processes=64, print_progress=True, sequential=False, async_return=False, desc='preload bytes for images')
        if self.have_msk:
            self.meta_msks_buffer = parallel_execution(self.meta_msks_path,
                                                   [self.orig_ixts[view_id]
                                                       for view_id in self.meta_viewids],
                                                   [self.Ds[view_id]
                                                       for view_id in self.meta_viewids],
                                                   [self.crop_h_w for i in range(
                                                       len(self.meta_viewids))],
                                                   [self.rs[view_id]
                                                       for view_id in (self.meta_viewids)],
                                                   '.png',
                                                   action=load_resize_undist_im_bytes, num_processes=64, print_progress=True, sequential=False, async_return=False, desc='preload bytes for masks')

    def read_img(self, frame_id, view_id):
        meta_idx = self.meta_idx[frame_id][view_id]
        if self.preload_data:
            img = load_image_from_bytes(
                self.meta_imgs_buffer[meta_idx], decode_flag=cv2.IMREAD_COLOR)[..., [2, 1, 0]]
            if self.have_msk:
                msk = load_image_from_bytes(
                self.meta_msks_buffer[meta_idx], decode_flag=cv2.IMREAD_GRAYSCALE)
            else: msk  = (np.ones_like(img[..., 0]) * 255).astype(np.uint8)
        else:
            img_path = self.meta_imgs_path[meta_idx]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            K, D, ratio = self.orig_ixts[view_id], self.Ds[view_id], self.rs[view_id]
            if np.linalg.norm(D) != 0.: img = cv2.undistort(img, K, D)
            if ratio != 1.: img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
            if self.have_msk:
                msk_path = self.meta_msks_path[meta_idx]
                msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                if np.linalg.norm(D) != 0.: msk = cv2.undistort(msk, K, D)
                if ratio != 1.: msk = cv2.resize(msk, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
            else:
                msk = (np.ones_like(img[..., 0]) * 255).astype(np.uint8)
            if self.crop_h_w[0] != -1 and self.crop_h_w[0] != img.shape[0]:
                assert(self.crop_h_w[0] < img.shape[0])
                crop = (img.shape[0] - self.crop_h_w[0]) // 2
                img = img[crop:crop+self.crop_h_w[0], ...]
                msk = msk[crop:crop+self.crop_h_w[0]]
            if self.crop_h_w[1] != -1 and self.crop_h_w[1] != img.shape[1]:
                assert(self.crop_h_w[1] < img.shape[1])
                crop = (img.shape[1] - self.crop_h_w[1]) // 2
                img = img[:, crop:crop+self.crop_h_w[1], ...]
                msk = msk[:, crop:crop+self.crop_h_w[1]]
            img = img[..., [2, 1, 0]]
        ###################
        # import matplotlib.pyplot as plt
        # plt.imsave('msk_orig.png', msk)
        # msk = (msk > 0.5).astype(np.uint8)
        # kernel = np.ones((5, 5), np.uint8)
        # plt.imsave('msk_01.png', msk)
        # msk = cv2.erode(msk, kernel)
        # plt.imsave('msk_erode.png', msk)
        # msk = cv2.dilate(msk, kernel)
        # plt.imsave('msk_dilate.png', msk)
        # msk = msk.astype(np.float32)
        # plt.imsave('msk_final.png', msk)
        ###################
        if 'schp' not in self.msk_dir: msk = (msk / 255.).astype(np.float32)
        else: msk = (msk != 0).astype(np.float32)
        img = (img / 255.).astype(np.float32)
        img = img * msk[..., None]
        if cfg.white_bkgd:
            img = img + 1 * (1 - msk[..., None])
        return img, msk

    def __getitem__(self, index):
        if self.split == 'train':
            batch_ids = np.random.choice(
                np.arange(len(self.meta)), self.imgs_per_batch, replace=False)
            ret = [self.get_batch(index) for index in batch_ids]
        else:
            ret = [self.get_batch(index)]
        return default_collate(ret)

    def sample_pixels(self, msk, frame_id, tar_view, h, w):
        num_pixels = cfg.num_pixels
        if self.bbox_type in ['RENBODY', 'NHR']:
            sample_msk = base_utils.get_bound_2d_mask(self.bounds[frame_id], self.ixts[tar_view], self.exts[tar_view][:3], h, w)
            select_inds = np.random.choice(sample_msk.reshape(-1).nonzero()[0], size=[num_pixels], replace=True)
            if cfg.get('num_patches', 0) >= 1: 
                patch_ratio = np.random.choice(cfg.patch_ratio)
                select_inds_ = PixelSampler.sample_patches(cfg.num_patches, cfg.patch_size, patch_ratio, h, w, (msk!=0).astype(np.uint8))
                select_inds_ = (select_inds_[1] * w + select_inds_[0]).astype(np.int32)
                select_inds = np.concatenate([select_inds, select_inds_], axis=0)
        else:
            select_inds = np.random.choice(h*w, size=[cfg.num_pixels], replace=True)
            sample_msk = None
            if cfg.num_patches >= 1: import ipdb; ipdb.set_trace()
        return select_inds, sample_msk

    def get_near_far(self, ext, ixt, h, w, frame_id):
        near_far = self.near_far.copy()
        # logger.debug('orig near_far: ' + str(near_far))
        if self.point_cloud is not None:
            near_far_ = get_near_far(self.point_cloud, np.linalg.inv(ext), ixt, h, w)
            near_far_[1] *= self.cfg.get('far_plane_ratio', 1.4)
            near_far_[0] *= self.cfg.get('near_plane_ratio', 0.8)
            near_far[0] = max(near_far[0], near_far_[0])
            near_far[1] = min(near_far[1], near_far_[1])
        # logger.debug('pcd near_far: ' + str(near_far))
        view_corners = self.corners[frame_id] @ ext[:3, :3].T + ext[:3, 3:].T
        near_far_ = np.array([view_corners[:, 2].min(), view_corners[:, 2].max()])
        near_far[0] = max(near_far[0], near_far_[0])
        near_far[1] = min(near_far[1], near_far_[1])
        # logger.debug('final near_far: ' + str(near_far))
        return near_far

    def get_batch(self, index):
        index = index % len(self.meta)
        frame_id, tar_view = self.meta[index][0], self.meta[index][1]
        if cfg.render_path: h, w = self.render_h_w
        else:
            rgb, msk = self.read_img(frame_id, tar_view)
            h, w = rgb.shape[:2]
        coords = self.coords[:h, :w].reshape(-1, 3).copy()
        if self.split == 'train':
            select_inds, sample_msk = self.sample_pixels(msk, frame_id, tar_view, h, w)
            coords = coords[select_inds]
            rgb, msk = rgb[coords[:, 1], coords[:, 0]
                           ], msk[coords[:, 1], coords[:, 0]]
        if cfg.render_path:
            rgb, msk = np.ones_like(coords[..., :3]), np.ones_like(coords[..., 0])
        else:
            rgb, msk = rgb.reshape(-1, 3), msk.reshape(-1)

        t = (frame_id - self.start_frame) / max((self.frame_len - 1),1)
        render_ext = self.render_exts[index] if cfg.render_path else self.exts[tar_view]
        render_ixt = self.render_ixts[index] if cfg.render_path else self.ixts[tar_view]
        render_ext_inv = self.render_exts_inv[index] if cfg.render_path else self.exts_inv[tar_view]
        render_ixt_inv = self.render_ixts_inv[index] if cfg.render_path else self.ixts_inv[tar_view]
        near_far = self.get_near_far(render_ext, render_ixt, h, w, frame_id)

        ret = {}
        if self.have_msk and 'enerf' in cfg.task and self.split == 'train': # compute tight cost volume
            ret.update({'mask_at_box': sample_msk.astype(np.uint8)})
        ret.update({'coords': coords, 'rgb': rgb, 'msk': msk})
        ret.update({'tar_ext': render_ext.astype(np.float32),
                    'tar_ixt': render_ixt.astype(np.float32),
                    'tar_ext_inv': render_ext_inv.astype(np.float32),
                    'tar_ixt_inv': render_ixt_inv.astype(np.float32),
                    'time': np.array(t).astype(np.float32),
                    'h_w': np.array([h, w]).astype(np.int32),
                    'near_far': near_far.astype(np.float32)})
        if self.have_msk: ret.update({'bounds': self.bounds[frame_id].astype(np.float32)})
        if self.have_msk and 'renbody' in self.data_root and self.cfg.get('use_global_bbox', True): ret.update({'bounds': self.bound.astype(np.float32)}) #
        ret.update({'meta': {'H': h, 'W': w, 'idx': index, 'frame_id': frame_id, 'view_id': tar_view}})
        return ret

    def __len__(self):
        return len(self.meta)
