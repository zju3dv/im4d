import torch
import torch.nn as nn
import imp

from lib.config.yacs import CfgNode as CN
from lib.config import logger


class UniformSampler(nn.Module):
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
        super(UniformSampler, self).__init__()
        self.num_samples = kwargs['num_samples']
        self.sample_level = kwargs['sample_level']
        
    
    def compute_t_vals(self, rays, **kwargs):
        sh = rays.shape[:-1]
        t_vals = torch.linspace(0., 1., steps=self.num_samples+1, device=rays.device)
        t_vals = t_vals.expand(sh + (self.num_samples+1,))
        if self.training:
            interval = 1/self.num_samples
            t_vals = torch.rand_like(t_vals) * interval + t_vals 
        return t_vals, {}
    
    def compute_z_vals(self, t_vals, rays, share_info, **kwargs):
        near, far = rays[..., 6:7], rays[..., 7:8]
        z_vals = near * (1. - t_vals) + far * t_vals
        return z_vals
        
    def forward(self, rays, **kwargs):
        rays_o, rays_d, near, far = rays[..., :3], rays[..., 3:6], rays[..., 6:7], rays[..., 7:8]
        t_vals, share_info = self.compute_t_vals(rays, **kwargs)
        z_vals = self.compute_z_vals(t_vals, rays, share_info, **kwargs)
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_mid[..., :, None]
        if 't2z_func' not in share_info:
            t2z_func = lambda t_vals, near, far: near * (1. - t_vals) + far * t_vals
            share_info.update({'t2z_func': t2z_func})
        share_info.update({'t_vals_{}'.format(self.sample_level-1): t_vals})# , 't2z_func': t2z_func})
        return points, z_vals, share_info