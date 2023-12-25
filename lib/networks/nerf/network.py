import torch
import torch.nn as nn
from torch.nn import functional as F
from os.path import join
import numpy as np
from lib.config import cfg, logger
if cfg.fast_render:
    import nerfacc
from lib.networks.nerf.encoders import get_encoder
from lib.networks.nerf.samplers import get_sampler
from lib.networks.nerf.fields import get_net
from lib.utils.ops.activations import init_density_activation
from lib.networks.nerf.activations import get_activation
from lib.utils.data_utils import get_ray_near_far_torch
from lib.utils.parallel_utils import chunkify, cat_dict, cat_tensor
from lib.utils.im4d.fastrend_utils import FastRendUtils
from lib.utils.im4d.im4d_utils import Im4DUtils
from lib.utils import data_utils

class Network(nn.Module):
    def __init__(self, net_cfg, sample_level=0):
        super(Network, self).__init__()
        # encoding
        self.net_cfg = net_cfg
        self.sample_level = sample_level
        self.only_geo = net_cfg.get('only_geo', False) # for some proposal network
        bbox = net_cfg.get('bounds') if 'bounds' in net_cfg else cfg.get('bounds')
        self.register_buffer('bbox', torch.tensor(bbox, dtype=torch.float32).reshape(2, 3))
        self.xyz_encoder = get_encoder(net_cfg.xyz_encoding)
        self.dir_encoder = get_encoder(net_cfg.dir_encoding)

        # proposal network
        self.sampler = get_sampler(cfg.samplers[sample_level], global_cfg=cfg, sample_level=sample_level)

        # radiance_network
        self.net = get_net(net_cfg.net,
                           xyz_in_dim = self.xyz_encoder.get_out_dim(),
                           view_in_dim = self.dir_encoder.get_out_dim() if net_cfg.get('dir_encoding', False) else None,
                           **net_cfg.net)
        # density activation
        self.density_act = init_density_activation(net_cfg.get('density_act', 'trunc_exp'))
        self.ignore_dist = net_cfg.get('ignore_dist', False) # for multi-view image-based rendering methods
        if cfg.fast_render and sample_level == 0: self.prepare_fast_render(net_cfg)
        
        logger.debug('network-level{}: xyz_encoder {}'.format(sample_level, str(self.xyz_encoder)))
        # logger.debug('network-level{}: dir_encoder {}'.format(sample_level, str(self.dir_encoder)))
        # logger.debug('network-level{}: decoder_net {}'.format(sample_level, str(self.net)))
        logger.debug('network-level{}: density_act {}'.format(sample_level, str(net_cfg.get('density_act', 'trunc_exp'))))
        logger.debug('network-level{}: ignore_dist {}'.format(sample_level, str(self.ignore_dist))) # Multi-view image-based rendering methods need to ignore the distance
        
    def prepare_fast_render(self, net_cfg):

        if cfg.get('separate', False):
            binarys_bg = np.load(join(cfg.grid_dir_bg, 'binarys.npz'))['arr_0'][0]
            bounds_bg = np.load(join(cfg.grid_dir_bg, 'bounds.npz'))['arr_0'][0]
            occ_grid = nerfacc.OccGridEstimator(torch.tensor(bounds_bg.reshape(-1)).float(), binarys_bg.shape)
            occ_grid.binaries = torch.tensor(binarys_bg).bool()[None]
            del occ_grid.grid_coords
            del occ_grid.grid_indices
            del occ_grid.occs
            occ_grid.cuda()
            self.occ_grid_bg = occ_grid
            self.render_step_size_bg = np.linalg.norm(bounds_bg[1] - bounds_bg[0]) / cfg.num_max_samples_bg

        binarys = np.load(join(cfg.grid_dir, 'binarys.npz'))['arr_0']
        bounds = np.load(join(cfg.grid_dir, 'bounds.npz'))['arr_0']
        self.occ_grids, self.render_step_size = {}, {}

        # NOTE: change it to `cfg.test_dataset.frame_sample` when testing
        frame_sample = cfg.train_dataset.frame_sample
        cnt = 0

        for frame_id in range(*frame_sample):
            occ_grid = nerfacc.OccGridEstimator(torch.tensor(bounds[cnt].reshape(-1)).float(), binarys[cnt].shape)
            occ_grid.binaries = torch.tensor(binarys[cnt]).bool()[None]

            del occ_grid.grid_coords
            del occ_grid.grid_indices
            del occ_grid.occs
            occ_grid.cuda()
            self.occ_grids[frame_id] = occ_grid
            self.render_step_size[frame_id] = np.linalg.norm(bounds[cnt][1] - bounds[cnt][0]) / cfg.num_max_samples

            cnt += 1

        if cfg.get('separate', False):
            logger.debug('background grid ' + str(binarys_bg.shape))
            logger.debug('background bounds ' + str(bounds_bg))
            logger.debug('background render_step_size ' + str(self.render_step_size_bg))
            logger.debug('grid ' + str(binarys[0].shape))
            logger.debug('bounds ' + str(bounds[0]))
            logger.debug('render_step_size ' + str(self.render_step_size[frame_id]))

    @staticmethod
    def volume_rendering(density, z_vals, ray_dir, act, ignore_dist=False):
        '''
        density: BxN_raysxN_samplesx1
        z_vals: Bx(N_rays+1)xN_samples
        '''
        if density.shape[2] == 1: return F.softmax(density, dim=-1)
        if act is not None: density = act(density)

        if ignore_dist: delta_density = density
        else:
            delta = z_vals[..., 1:] - z_vals[..., :-1]
            delta = delta * ray_dir.norm(dim=-1, keepdim=True)
            delta_mask = delta > 1e-6
            delta = delta[delta_mask]

            delta_density = torch.zeros_like(density)
            delta_density[delta_mask] = delta * density[delta_mask]

        alpha = 1 - torch.exp(-delta_density)

        T = torch.cumprod(1 - alpha + 1e-10, dim=-1)[..., :-1]
        T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
        weights = alpha * T
        return weights
    
    @staticmethod
    def compute_ndc_rays(rays_o, rays_d, near_far, focal, H, W, near=1.):
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        rays_o = rays_o + t[...,None] * rays_d

        o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
        o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
        o2 = 1. + 2. * near / rays_o[...,2]
        
        d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        d2 = -2. * near / rays_o[...,2]

        rays_o = torch.stack([o0,o1,o2], -1)
        rays_d = torch.stack([d0,d1,d2], -1)
        # rays_d = rays_d / np.linalg.norm(rays_d, axis=-1)[..., None]
        rays_d = rays_d / 2.
        near_far[..., 0] = 0. 
        near_far[..., 1] = 1.99 
        return rays_o, rays_d, near_far
        
    @staticmethod
    def compute_rays(coords, ext_inv, ixt_inv, time, h_w, near_far):
        rays_o = ext_inv[:, :3, 3][:, None, :].repeat(1, coords.shape[1], 1)
        rays_d = coords.float() @ ixt_inv.transpose(-1, -2) @ ext_inv[:, :3, :3].transpose(-1, -2)
        near_far = near_far[:, None, :].repeat(1, coords.shape[1], 1)
        rays_t = time[:, None, None].repeat(1, coords.shape[1], 1)
        uv = coords[:, :, :2] / (h_w[..., [1, 0]][:, None, :] - 1)
        if cfg.get('ndc', False): rays_o, rays_d, near_far = Network.compute_ndc_rays(rays_o, rays_d, near_far, cfg.ndc_FOCAL, cfg.ndc_H, cfg.ndc_W)
        return torch.cat([rays_o, rays_d, near_far, rays_t, uv], dim=-1)

    @staticmethod
    def prefilter_rays(batch, rays):
        assert(len(rays) == 1)
        _near, _far, mask_at_box = get_ray_near_far_torch(batch['bounds'][0], rays[0, :, :3], rays[0, :, 3:6])
        rays = rays[0][mask_at_box][None]
        batch.update({'mask_at_box': mask_at_box[None].to(torch.uint8),
                      'rgb': batch['rgb'][0][mask_at_box][None],
                      'msk': batch['msk'][0][mask_at_box][None]})
        return rays

    def render_rays(self, rays, **kwargs):
        '''
        rays: BxN_raysx8
        '''
        ray_dir = rays[..., 3:6]
        points, z_vals, ret_info = self.sampler(rays, **kwargs)
        # BxN_raysxN_samplesxC

        viewdir = ray_dir / ray_dir.norm(dim=-1, keepdim=True)
        time = rays[..., 8:9][:, :, None].repeat(1, 1, points.shape[-2], 1)
        xyz_encoding = self.xyz_encoder(points, viewdir=viewdir, sample_level=self.sample_level, bbox=self.bbox, time=time, **kwargs) # feature encoding may need viewdir

        if self.only_geo:  _, sigma = self.net(xyz_encoding, None, only_geo=True)
        else:
            dir_encoding = self.dir_encoder(viewdir)
            dir_encoding = dir_encoding[..., None, :].repeat(1, 1, points.shape[-2], 1) if dir_encoding is not None else None
            rgb, sigma = self.net(xyz_encoding, dir_encoding)

        vr_weights = self.volume_rendering(density=sigma[..., 0],
                                           z_vals=z_vals,
                                           ray_dir=ray_dir,
                                           act=self.density_act,
                                           ignore_dist=self.ignore_dist)

        output = {}
        for k in ret_info:
            if 'weights' in k and not self.training: continue # do not need weights in testing
            output.update({k: ret_info[k]})

        acc_map = vr_weights.sum(dim=-1)
        if cfg.depth_method == 'expected': depth_map = (vr_weights * (z_vals[..., :-1] + z_vals[..., 1:]) * 0.5).sum(dim=-1) # / (acc_map + 1e-6)
        elif cfg.depth_method == 'median': 
            depth_map = Im4DUtils.render_median_depth(vr_weights, z_vals)
            pts = rays[..., :3] + depth_map[..., None] * ray_dir
            output.update({'pts_{}'.format(self.sample_level): pts})
        else: import ipdb; ipdb.set_trace()
        # TODO: median depth
        output.update({
            'depth_map_{}'.format(self.sample_level): depth_map,
            'acc_map_{}'.format(self.sample_level): acc_map,
            'z_vals_{}'.format(self.sample_level): z_vals,
        })
        if not self.only_geo:
            # if rgb.shape[-1] == 6: rgb_map = torch.cat([(vr_weights[..., None] * rgb[..., :3]).sum(dim=-2), (vr_weights[..., None] * rgb[..., 3:]).sum(dim=-2)], dim=-1)
            # else: rgb_map = (vr_weights[..., None] * rgb).sum(dim=-2)
            rgb_map = (vr_weights[..., None] * rgb).sum(dim=-2)
            if cfg.white_bkgd: rgb_map = rgb_map + (1-acc_map[..., None])
            output.update({'rgb_map_{}'.format(self.sample_level): rgb_map})

        if self.training or self.sample_level > 0: output.update({'weights_{}'.format(self.sample_level): vr_weights})
        # include the weights for training or sampling at a higher level
        return output

    def render_rays_fast(self, rays, **kwargs):
        # only support for Im4D
        assert rays.shape[0] == 1
        # assert cfg.task == 'im4d'
        rays_o, rays_d = rays[0, :, :3], rays[0, :, 3:6]
        viewdir = rays_d / rays_d.norm(dim=-1, keepdim=True)
        n_rays = len(rays_o)
        time = rays[0, :, 8:9]
        frame_id = kwargs['batch']['meta']['frame_id'].item() # batch_size = 1

        # raymarching
        ray_indices, t_starts, t_ends = self.occ_grids[frame_id].sampling(
            rays_o,
            rays_d,
            render_step_size=self.render_step_size[frame_id],
            stratified=self.training
        )

        if self.training and cfg.get('separate', False) and cfg.use_fg_msk:
            t_mid = (t_starts + t_ends) / 2.0
            points = rays_o[ray_indices] + t_mid[:, None] * rays_d[ray_indices]
            v = viewdir[ray_indices]
            t = time[0].item() * torch.ones_like(v[..., :1])
            xyz_encoding = self.xyz_encoder(points[None, :, None], only_geo=True, sample_level=self.sample_level, bbox=self.bbox, time=t[None, :, None], **kwargs)
            _, density = self.net([xyz_encoding], None, only_geo=True)
            density = self.density_act(density)
            delta_density = density * self.render_step_size[frame_id] * rays_d[ray_indices].norm(dim=-1)[None, :, None, None]
            alpha = 1 - torch.exp(-delta_density)
            weights, trans = nerfacc.render_weight_from_alpha(alpha[0, :, 0, 0], ray_indices=ray_indices, n_rays=n_rays)

            fg_acc_map = nerfacc.accumulate_along_rays(
                weights=weights,
                values=None,
                ray_indices=ray_indices,
                n_rays=n_rays,
            )[None, :, 0]

        if cfg.get('separate', False):
            ray_indices_bg, t_starts_bg, t_ends_bg = self.occ_grid_bg.sampling(
                rays_o,
                rays_d,
                render_step_size=self.render_step_size_bg,
                stratified=self.training
            )

            ray_indices = torch.concat((ray_indices, ray_indices_bg), dim=0)
            t_starts = torch.concat((t_starts, t_starts_bg), dim=0)
            t_ends = torch.concat((t_ends, t_ends_bg), dim=0)

            ray_indices, idx = torch.sort(ray_indices)
            t_starts = t_starts[idx]
            t_ends = t_ends[idx]

        if len(ray_indices) == 0:
            acc_map = torch.zeros_like(rays[..., 0])
            depth_map = torch.zeros_like(rays[..., 0])
            rgb_map = torch.zeros_like(rays[..., :3])
            if cfg.white_bkgd: rgb_map = rgb_map + (1-acc_map[..., None])
            output = {}
            output.update({
                'depth_map_{}'.format(self.sample_level): depth_map,
                'acc_map_{}'.format(self.sample_level): acc_map,
                'rgb_map_{}'.format(self.sample_level): rgb_map
            })
            return output

        # compute vr weights
        t_mid = (t_starts + t_ends) / 2.0
        points = rays_o[ray_indices] + t_mid[:, None] * rays_d[ray_indices]
        v = viewdir[ray_indices]
        t = time[0].item() * torch.ones_like(v[..., :1])
        xyz_encoding = self.xyz_encoder(points[None, :, None], only_geo=True, sample_level=self.sample_level, bbox=self.bbox, time=t[None, :, None], **kwargs)
        _, density = self.net([xyz_encoding], None, only_geo=True)
        density = self.density_act(density)
        delta_density = density * self.render_step_size[frame_id] * rays_d[ray_indices].norm(dim=-1)[None, :, None, None]
        alpha = 1 - torch.exp(-delta_density)
        weights, trans = nerfacc.render_weight_from_alpha(alpha[0, :, 0, 0], ray_indices=ray_indices, n_rays=n_rays)

        # filter weight_threshold
        if cfg.get('separate', False):
            rgb_msk = torch.ones_like(weights).bool()
        else:
            rgb_msk = weights >= cfg.weight_thresh # only compute rgb for weighted samples
            if rgb_msk.sum() == 0:
                acc_map = torch.zeros_like(rays[..., 0])
                depth_map = torch.zeros_like(rays[..., 0])
                rgb_map = torch.zeros_like(rays[..., :3])
                if cfg.white_bkgd: rgb_map = rgb_map + (1-acc_map[..., None])
                output = {}
                output.update({
                    'depth_map_{}'.format(self.sample_level): depth_map,
                    'acc_map_{}'.format(self.sample_level): acc_map,
                    'rgb_map_{}'.format(self.sample_level): rgb_map
                })
                return output

        # compute rgb fields
        ibr_feat = self.xyz_encoder.feature_projection(points[rgb_msk][None, :, None], viewdir=v[rgb_msk][None, :, None], time=t[rgb_msk][None, :, None], sample_level=self.sample_level, **kwargs)
        rgb, _ = self.net.enerf_net(ibr_feat, v[rgb_msk][None, :, None][..., :3])

        rgb_map = nerfacc.accumulate_along_rays(
            weights=weights[rgb_msk],
            values=rgb[0, :, 0, :],
            ray_indices=ray_indices[rgb_msk],
            n_rays=n_rays,
        )[None, :]
        acc_map = nerfacc.accumulate_along_rays(
            weights=weights[rgb_msk],
            values=None,
            ray_indices=ray_indices[rgb_msk],
            n_rays=n_rays,
        )[None, :, 0]
        depth_map = nerfacc.accumulate_along_rays(
            weights=weights[rgb_msk],
            values=t_mid[rgb_msk][:, None],
            ray_indices=ray_indices[rgb_msk],
            n_rays=n_rays,
        )[None, :, 0]

        if cfg.white_bkgd: rgb_map = rgb_map + (1 - acc_map[..., None])
        output = {}
        output.update({
            'depth_map_{}'.format(self.sample_level): depth_map,
            'acc_map_{}'.format(self.sample_level): acc_map,
            'rgb_map_{}'.format(self.sample_level): rgb_map
        })
        if self.training and cfg.get('separate', False) and cfg.use_fg_msk:
            output.update({
                'fg_acc_map_{}'.format(self.sample_level): fg_acc_map
            })
        return output

    def forward(self, batch, batch_share_info={}):
        rays = self.compute_rays(batch['coords'], batch['tar_ext_inv'], batch['tar_ixt_inv'], batch['time'], batch['h_w'], batch['near_far'])
        if not self.training and 'bounds' in batch: rays = self.prefilter_rays(batch, rays)
        render_rays_func = self.render_rays_fast if cfg.fast_render else self.render_rays
        ret = chunkify(render_rays_func, cat_dict, [rays], chunk_dim=1, chunk_size=cfg.chunk_size, batch=batch, batch_share_info=batch_share_info)
        ret.update({'model': self}) # for regularization
        return ret

    def inference_density(self, wpts, **kwargs):
        wpts = wpts[None, :, None]
        time = torch.ones_like(wpts[..., :1]) * kwargs['batch']['time'][0]
        xyz_encoding = self.xyz_encoder(wpts, only_geo=True, sample_level=self.sample_level, bbox=self.bbox, time=time, **kwargs)
        if cfg.task == 'im4d':
            _, sigma = self.net([xyz_encoding], None, only_geo=True)
        else:
            _, sigma = self.net(xyz_encoding, None, only_geo=True)
        return sigma.reshape(-1, 1)

    def cache_grid(self, batch, save_ply=False):
        assert(len(batch['bounds']) == 1)
        wpts = FastRendUtils.prepare_wpts(batch['bounds'][0].cpu(), cfg.grid_resolution, cfg.subgrid_resolution)
        # wpts = data_utils.get_space_points_torch(batch['bounds'][0], cfg.grid_resolution)
        shape = wpts.shape

        binary = torch.zeros(cfg.grid_resolution)
        chunk_size_x = cfg.chunk_size_x
        chunk_size_y = cfg.chunk_size_y
        for i in range(0, shape[0], chunk_size_x):
            for j in range(0, shape[1], chunk_size_y):
                sigma = self.inference_density(wpts[i:i + chunk_size_x, j:j + chunk_size_y].reshape(-1, 3).cuda(), batch=batch)
                sigma = sigma.reshape(chunk_size_x, chunk_size_y, shape[2], shape[3])
                binary[i:i + chunk_size_x, j:j + chunk_size_y] = (sigma > cfg.sigma_thresh).float().sum(dim=-1) > 0

        if cfg.padding_grid:
            structuring_element = torch.ones(1, 1, 3, 3, 3, device=binary.device)
            dilated_tensor = F.conv3d(binary[None, None].float(), structuring_element.float(), padding=1) > 0
            binary = dilated_tensor[0, 0].float()

        if save_ply: data_utils.save_mesh_with_extracted_fields(join(cfg.grid_dir, 'meshes', '{:06d}.ply'.format(batch['meta']['frame_id'][0].item())), binary.cpu().numpy(), batch['bounds'][0].cpu().numpy(), grid_resolution=cfg.grid_resolution)
        
        return binary.cpu().numpy().astype(bool), batch['bounds'][0].cpu().numpy()