import numpy as np
import cv2
import os
from os.path import join
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

from lib.config import cfg, logger
from lib.datasets.volcap.base_dataset import Dataset as BaseDataset
from lib.utils.data_utils import load_path_pose


class Dataset(BaseDataset):
    
    def get_render_path_meta(self, input_views):
        if self.path_type == 'NERFSTUDIO':
            repeat_num = 10
            path_path = join(self.data_root, 'nerfstudio/camera_paths', self.path_name)
            pcd_path = join(self.data_root, 'nerfstudio/exports/pcd', self.cfg.get('scene_pcd'))
            if not os.path.exists(path_path): path_path = join(self.data_root, self.path_name)
            if not os.path.exists(pcd_path): pcd_path = join(self.data_root, self.cfg.get('scene_pcd'))
            poses, ixts, near_fars, h, w = load_path_pose(path_path, pcd_path)

            poses = np.concatenate([poses if i%2==0 else poses[::-1] for i in range(repeat_num)])
            ixts = np.concatenate([ixts if i%2==0 else ixts[::-1] for i in range(repeat_num)])
            near_fars = np.concatenate([near_fars if i%2==0 else near_fars[::-1] for i in range(repeat_num)])

            self.render_exts = poses
            self.render_ixts = ixts
            self.render_near_fars = near_fars
            self.render_h_w = [h, w]
            self.render_ratio = self.cfg.get('render_ratio', 1.)
            if self.render_ratio != 1.:
                self.render_ixts[:, :2] *= self.render_ratio
                self.render_h_w = [int(h * self.render_ratio), int(w * self.render_ratio)]
            render_crop = self.cfg.get('render_crop', [-1, -1])
            orig_h, orig_w = self.render_h_w
            render_h, render_w = render_crop
            if render_h != -1 and orig_h > render_h:
                self.render_h_w[0] = render_h
                self.render_ixts[:, 1, 2] -= (orig_h - render_h) // 2
            if render_w != -1 and orig_w > render_w:
                self.render_h_w[1] = render_w
                self.render_ixts[:, 0, 2] -= (orig_w - render_w) // 2
            h, w = self.render_h_w
            if h%32 != 0: h = h//32 * 32
            if w%32 != 0: w = w//32 * 32
            self.render_h_w = [h, w]
            self.render_exts_inv = np.linalg.inv(self.render_exts).astype(np.float32)
            self.render_ixts_inv = np.linalg.inv(self.render_ixts).astype(np.float32)

            stop_frames = {} # 
            frames = []
            frames_ = np.arange(*self.cfg.get('frame_sample'))
            for frame_id in frames_:
                if frame_id in stop_frames:
                    frames += [frame_id] * stop_frames[frame_id]
                else:
                    frames.append(frame_id)

            # poses = poses[:len(frames)]
            if self.frame_len == 1:
                frame_id = self.cfg.get('frame_sample')[0]
                # frame_id = kwargs['frame_sample'][0]
                for ext in poses:
                    near_views = self.get_near_views(np.linalg.inv(ext), input_views)
                    for idx in near_views:
                        self.preload_img(frame_id, idx)
                    self.meta += [(frame_id, -1, near_views)]
            else:
                # frames = np.arange(*self.cfg.get('frame_sample'))[:len(poses)]
                for ext, frame_id in zip(poses, frames):
                    near_views = self.get_near_views(np.linalg.inv(ext), input_views)
                    for idx in near_views:
                        self.preload_img(frame_id, idx)
                    self.meta += [(frame_id, -1, near_views)]

            return

        super().get_render_path_meta(input_views)
        for index, meta in enumerate(self.meta):
            render_ext = self.render_exts[index]
            near_views = self.get_near_views(np.linalg.inv(render_ext), input_views)
            self.meta[index] = (meta[0], -1, near_views)
            for idx in near_views:
                self.preload_img(meta[0], idx)

    def get_near_views(self, c2w, input_views, remove_view=-1):
        input_cam_xyz = np.array([self.exts_inv[view_id][:3, 3]
                                 for view_id in input_views])
        render_cam_xyz = c2w[:3, 3]
        spatial_distances = np.linalg.norm(
            input_cam_xyz - render_cam_xyz[None], axis=-1)
        
        input_cam_rot = np.array([self.exts_inv[view_id][:3, :3] for view_id in input_views])
        render_cam_rot = c2w[:3, :3]
        rot_distances = np.linalg.norm(input_cam_rot - render_cam_rot[None], axis=(-2, -1))
        argsorts = np.argsort(spatial_distances)
        argsorts_rot = np.argsort(rot_distances)
        max_input_views = self.cfg.get('train_input_views')[-1] + 2 if self.split == 'train' else self.cfg.get('test_input_views')
        rot_num = max_input_views // 2
        spatial_num = max_input_views - rot_num
        if remove_view in input_views:
            return [input_views[i] for i in argsorts[1:max_input_views+1]]
        else:
            return [input_views[i] for i in argsorts[1:max_input_views+1]]
            ret_views = [input_views[i] for i in argsorts[:spatial_num]] + [input_views[i] for i in argsorts_rot[:rot_num]]
            # # may repeat, then remove the repeated and add new ones acoording to rot_distances
            # ret_views = list(set(ret_views))
            # while len(ret_views) < max_input_views:
            #     for i in range(len(input_views)):
            #         if input_views[i] not in ret_views:
            #             ret_views.append(input_views[i])
            #             break
            # print(ret_views)
            # return ret_views
                

    def get_metas(self, render_views, input_views):
        logger.info('input views: ' + str(input_views))
        if self.render_path:
            self.get_render_path_meta(input_views)
            return
        for frame_id in tqdm(range(*self.cfg.get('frame_sample')), desc='loading meta'):
            for view_id in render_views:
                self.preload_img(frame_id, view_id)
                near_views = self.get_near_views(
                    self.exts_inv[view_id], input_views, remove_view=view_id)
                for idx in near_views:
                    self.preload_img(frame_id, idx)
                self.meta += [(frame_id, view_id, near_views)]

    def __getitem__(self, index):
        if self.split == 'train':
            batch_ids = np.random.choice(
                np.arange(len(self.meta)), self.imgs_per_batch, replace=False)
            num_input_views = np.random.choice(self.cfg.get(
                'train_input_views'), p=self.cfg.get('train_input_views_prob'))
            ret = [self.get_batch(index, num_input_views=num_input_views)
                   for index in batch_ids]
            ret = self.padding_srcinps(ret)
        else:
            ret = [self.get_batch(
                index, num_input_views=self.cfg.get('test_input_views'))]
        return default_collate(ret)

    def padding_srcinps(self, ret):
        hs = [batch['src_inps'].shape[2] for batch in ret]
        ws = [batch['src_inps'].shape[3] for batch in ret]
        H, W = np.max(hs), np.max(ws)
        for idx, batch in enumerate(ret):
            h, w = hs[idx], ws[idx]
            if h != H or w != W:
                src_inps = np.ones((batch['src_inps'].shape[0], batch['src_inps'].shape[1], H, W), dtype=np.float32) if cfg.white_bkgd else np.zeros(
                    (batch['src_inps'].shape[0], batch['src_inps'].shape[1], H, W), dtype=np.float32) - 1.
                crop_h, crop_w = (H - h)//2, (W - w)//2
                src_inps[:, :, crop_h:crop_h+h,
                         crop_w:crop_w+w] = batch['src_inps']
                batch['src_inps'] = src_inps
                batch['src_ixts'][:, 0, 2] += crop_w
                batch['src_ixts'][:, 1, 2] += crop_h
        return ret

    def get_batch(self, index, num_input_views):
        ret = super().get_batch(index)
        frame_id, tar_view, near_views = self.meta[index]
        if self.split == 'train':
            if np.random.random() <= 0.05:
                near_views = [tar_view] + near_views
            src_views = np.random.choice(
                near_views[:num_input_views+2], num_input_views, replace=False)
        else:
            src_views = near_views[:num_input_views]
        src_inps, src_exts, src_ixts = self.read_srcs(frame_id, src_views)
        ret.update({'src_inps': src_inps, 'src_exts': src_exts, 'src_ixts': src_ixts})
        return ret

    def read_srcs(self, frame_id, src_views):
        if not self.cfg.get('crop_srcinps', False):
            src_inps = (np.asarray([self.read_img(frame_id, view_id)[
                        0] for view_id in src_views]).transpose(0, 3, 1, 2)).astype(np.float32) * 2. - 1.
            src_exts = self.exts[src_views]
            src_ixts = self.ixts[src_views]
            return src_inps, src_exts, src_ixts
        src_imgs, src_ixts = [], []
        ws, hs = [], []
        for view_id in src_views:
            img, msk = self.read_img(frame_id, view_id)
            if cfg.get('separate', False):
                msk = (np.ones_like(img[..., 0])).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(msk.astype(np.uint8))
            ws.append(w)
            hs.append(h)
            src_imgs.append((img[y:y+h, x:x+w]).astype(np.float32))
            src_ixt = self.ixts[view_id].copy()
            src_ixt[0, 2] -= x
            src_ixt[1, 2] -= y
            src_ixts.append(src_ixt)
        src_exts = self.exts[src_views]
        src_ixts = np.array(src_ixts)
        W, H = (np.max(ws) // 8 + 1) * 8, (np.max(hs) // 8 + 1) * 8
        # padding src_imgs to the same shape (H, W)
        src_inps = []
        for idx, src_img in enumerate(src_imgs):
            src_inp = np.ones((H, W, 3), dtype=np.float32) if cfg.white_bkgd else np.zeros(
                (H, W, 3), dtype=np.float32)
            h, w = src_img.shape[:2]
            crop_h, crop_w = (H - h)//2, (W - w)//2
            src_inp[crop_h:crop_h+h, crop_w:crop_w+w] = src_img
            src_ixts[idx][0, 2] += crop_w
            src_ixts[idx][1, 2] += crop_h
            src_inps.append(src_inp)
        src_inps = np.array(src_inps).transpose(0, 3, 1, 2) * 2 - 1
        return src_inps, src_exts, src_ixts
