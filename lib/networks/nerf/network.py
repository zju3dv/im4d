import torch
import torch.nn as nn
from torch.nn import functional as F
from os.path import join
import numpy as np
from lib.config import cfg
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

    def prepare_fast_render(self, net_cfg):
        binarys = np.load(join(cfg.grid_dir, 'binarys.npz'))['arr_0']
        bounds = np.load(join(cfg.grid_dir, 'bounds.npz'))['arr_0']
        self.occ_grids, self.render_step_size = {}, {}
        for frame_id in range(*cfg.test_dataset.frame_sample):
            occ_grid = nerfacc.OccGridEstimator(torch.tensor(bounds[frame_id].reshape(-1)).float(), binarys[frame_id].shape)
            occ_grid.binaries = torch.tensor(binarys[frame_id]).bool()[None]
            del occ_grid.grid_coords
            del occ_grid.grid_indices
            del occ_grid.occs
            occ_grid.cuda()
            self.occ_grids[frame_id] = occ_grid
            self.render_step_size[frame_id] = np.linalg.norm(bounds[frame_id][1] - bounds[frame_id][0]) / cfg.num_max_samples

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
    def compute_rays(coords, ext_inv, ixt_inv, time, h_w, near_far):
        rays_o = ext_inv[:, :3, 3][:, None, :].repeat(1, coords.shape[1], 1)
        rays_d = coords.float() @ ixt_inv.transpose(-1, -2) @ ext_inv[:, :3, :3].transpose(-1, -2)
        near_far = near_far[:, None, :].repeat(1, coords.shape[1], 1)
        rays_t = time[:, None, None].repeat(1, coords.shape[1], 1)
        uv = coords[:, :, :2] / (h_w[..., [1, 0]][:, None, :] - 1)
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
            rgb_map = (vr_weights[..., None] * rgb).sum(dim=-2)
            if cfg.white_bkgd: rgb_map = rgb_map + (1-acc_map[..., None])
            output.update({'rgb_map_{}'.format(self.sample_level): rgb_map})

        if self.training or self.sample_level > 0: output.update({'weights_{}'.format(self.sample_level): vr_weights})
        # include the weights for training or sampling at a higher level
        return output

    def render_rays_fast(self, rays, **kwargs):
        # only support for Im4D
        assert cfg.task == 'im4d'
        rays_o, rays_d = rays[0, :, :3], rays[0, :, 3:6]
        viewdir = rays_d / rays_d.norm(dim=-1, keepdim=True)
        n_rays = len(rays_o)
        time = rays[0, :, 8:9]
        frame_id = kwargs['batch']['meta']['frame_id'].item() # batch_size = 1

        # raymarching
        ray_indices, t_starts, t_ends = self.occ_grids[frame_id].sampling(
            rays_o,
            rays_d,
            render_step_size=self.render_step_size[frame_id]
        )
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

        if cfg.white_bkgd: rgb_map = rgb_map + (1-acc_map[..., None])
        output = {}
        output.update({
            'depth_map_{}'.format(self.sample_level): depth_map,
            'acc_map_{}'.format(self.sample_level): acc_map,
            'rgb_map_{}'.format(self.sample_level): rgb_map
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
        _, sigma = self.net([xyz_encoding], None, only_geo=True)
        return sigma.reshape(-1, 1)

    def cache_grid(self, batch, save_ply=False):
        assert(len(batch['bounds']) == 1)
        wpts = FastRendUtils.prepare_wpts(batch['bounds'][0], cfg.grid_resolution, cfg.subgrid_resolution)
        # wpts = data_utils.get_space_points_torch(batch['bounds'][0], cfg.grid_resolution)
        shape = wpts.shape
        sigma = chunkify(self.inference_density, cat_tensor, [wpts.reshape(-1, 3)], chunk_size=cfg.chunk_size, chunk_dim=0, batch=batch)
        sigma = sigma.reshape(shape[:-1])
        binary = (sigma > cfg.sigma_thresh).float().sum(dim=-1) > 0
        if cfg.padding_grid:
            structuring_element = torch.ones(1, 1, 3, 3, 3, device=binary.device)
            dilated_tensor = F.conv3d(binary[None, None].float(), structuring_element.float(), padding=1) > 0
            binary = dilated_tensor[0, 0].float()
        if save_ply: data_utils.save_mesh_with_extracted_fields(join(cfg.grid_dir, 'meshes', '{:06d}.ply'.format(batch['meta']['frame_id'][0].item())), binary.cpu().numpy(), batch['bounds'][0].cpu().numpy(), grid_resolution=cfg.grid_resolution)
        return binary.cpu().numpy().astype(np.bool), batch['bounds'][0].cpu().numpy()

