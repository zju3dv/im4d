import torch
import torch.nn as nn
import torch.nn.functional as F
import imp
import numpy as np
from lib.networks.nerf.samplers.utils import pdf_sample
from lib.networks.nerf.samplers.uniform_sampler import UniformSampler

from lib.config.yacs import CfgNode as CN


class MvsSampler(nn.Module):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        global_cfg=None,
        **kwargs,
    ) -> None:
        super(MvsSampler, self).__init__()
        self.num_samples = kwargs['num_samples']
        
        
    def compute_t_vals(self, rays, **kwargs):
        sh = rays.shape[:-1]
        t_vals = torch.linspace(0., 1., steps=self.num_samples+1, device=rays.device)
        t_vals = t_vals.expand(sh + (self.num_samples+1,))
        return t_vals, {}
    
    def compute_z_vals(self, t_vals, rays, share_info, **kwargs):
        near, far = rays[..., 6:7], rays[..., 7:8]
        z_vals = near * (1. - t_vals) + far * t_vals
        return z_vals
        
    def forward(self, rays, **kwargs):
        t_mean_std_min_max = kwargs['batch_share_info']['t_mean_std_min_max']
        t2z_func = kwargs['batch_share_info']['t2z_func']
        uv = rays[..., -2:]
        t_h_w = t_mean_std_min_max.shape[2:]
        render_h_w = [kwargs['batch']['meta']['H'][0].item(), kwargs['batch']['meta']['W'][0].item()]
        if t_h_w[0] == render_h_w[0] and t_h_w[1] == render_h_w[1]:
            x, y = (uv[..., 0] * (t_h_w[1] - 1)).long(), (uv[..., 1] * (t_h_w[0] -1 )).long()
            ray_near_far = t_mean_std_min_max[:, :, y, x][:, :, 0].transpose(-1, -2)
        else:
            ray_near_far = F.grid_sample(t_mean_std_min_max, uv[:, None]*2-1., align_corners=True, mode='bilinear')[:, :, 0].transpose(-1, -2)
        uniform_vals, share_info = self.compute_t_vals(rays, **kwargs)
        t_vals = ray_near_far[..., 2:3] * (1 - uniform_vals) + ray_near_far[..., 3:4] * uniform_vals
        B, num_rays, num_samples = t_vals.shape
        z_vals = t2z_func(t_vals.reshape(B, -1), kwargs['batch']['near_far']).reshape(B, num_rays, num_samples)
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_mid[..., :, None]
        # t2z_func = lambda t_vals, near, far: near * (1. - t_vals) + far * t_vals
        share_info.update({'t_vals': t_vals})
        return points, z_vals, share_info