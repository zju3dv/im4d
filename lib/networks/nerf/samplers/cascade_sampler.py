import torch
import torch.nn as nn
import imp
import numpy as np
from lib.networks.nerf.samplers.utils import pdf_sample
from lib.networks.nerf.samplers.uniform_sampler import UniformSampler

from lib.config.yacs import CfgNode as CN


class CascadeSampler(UniformSampler):
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
        super(CascadeSampler, self).__init__(**kwargs)
        module = global_cfg.network_module
        path = global_cfg.network_path
        self.network = imp.load_source(module, path).Network(CN(kwargs), sample_level=kwargs['sample_level'])
        self.sample_level = kwargs['sample_level']
        self.num_samples = kwargs['num_samples']
        
        self.N_anneal_steps = kwargs['N_anneal_steps']
        self.N_anneal_slope = kwargs['N_anneal_slope']
        self.bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
        self.include_last_level = kwargs.get('include_input', False)
        
    def compute_t_vals(self, rays, **kwargs):
        ret_info = self.network.render_rays(rays, **kwargs)
        pdf_weights = ret_info['weights_{}'.format(self.sample_level)]
        if self.training:
            assert('step' in kwargs['batch'])
            step = kwargs['batch']['step']
            train_frac = np.clip(step / self.N_anneal_steps, 0, 1)
            anneal = self.bias(train_frac, self.N_anneal_slope)
            pdf_weights = torch.pow(pdf_weights, anneal)
        with torch.no_grad():
            t_vals = pdf_sample(ret_info[f't_vals_{self.sample_level}'], pdf_weights, self.training, self.num_samples)
        if self.include_last_level:
            t_vals = torch.cat([t_vals, ret_info[f't_vals_{self.sample_level}'][..., 1:]], dim=-1)
            t_vals = torch.sort(t_vals, dim=-1)[0]
        return t_vals, ret_info
    
    def compute_z_vals(self, t_vals, rays, share_info, **kwargs):
        near, far = rays[..., 6:7], rays[..., 7:8]
        t2z_func = share_info[f't2z_func']
        return t2z_func(t_vals, near, far)