import torch
import torch.nn as nn
import imp
from lib.networks.nerf.samplers.uniform_sampler import UniformSampler

from lib.config.yacs import CfgNode as CN
from lib.config import logger


class DispUniformSampler(UniformSampler):
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
        super(DispUniformSampler, self).__init__(**kwargs)
        self.num_samples = kwargs['num_samples']
        self.sample_level = kwargs['sample_level']
    
    def compute_z_vals(self, t_vals, rays, share_info, **kwargs):
        near, far = rays[..., 6:7], rays[..., 7:8]
        z_vals_inv = 1/near * (1. - t_vals) + 1/far * t_vals
        z_vals = 1/z_vals_inv
        t2z_func = lambda t_vals, near, far: 1/(1/near * (1. - t_vals) + 1/far * t_vals)
        share_info.update({'t2z_func': t2z_func})
        return z_vals