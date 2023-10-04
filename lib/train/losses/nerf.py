import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg
import math
EPS = 1.0e-7
from lib.train.losses.vgg_perceptual_loss import VGGPerceptualLoss

def outer(
    t0_starts: torch.Tensor,
    t0_ends: torch.Tensor,
    t1_starts: torch.Tensor,
    t1_ends: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    """Faster version of
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64
    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), right=True) - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), right=True)
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    # t0 = torch.cat([t0_starts, t0_ends[..., -1:]], dim=-1)
    # t1 = torch.cat([t1_starts, t1_ends[..., -1:]], dim=-1)
    # i = torch.arange(t1.shape[-1], device=t1.device)
    # t0_ge_t1 = t0[..., None, :] >= t1[..., :, None]
    # idx_lo = torch.where(t0_ge_t1, i[..., :, None], i[..., :1, None]).max(dim=-2).values
    # idx_hi = torch.where(~t0_ge_t1, i[..., :, None], i[..., -1:, None]).min(dim=-2).values
    # cy1_lo = torch.take_along_dim(cy1, idx_lo, dim=-1)
    # cy1_hi = torch.take_along_dim(cy1, idx_hi, dim=-1)
    # y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]

    return y0_outer

def lossfun_outer(
    t: torch.Tensor,  # [..., "num_samples+1"],
    w: torch.Tensor,  # [..., "num_samples"],
    t_env: torch.Tensor,  # [..., "num_samples+1"],
    w_env: torch.Tensor,  # [..., "num_samples"],
):
    """
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L136
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L80
    Args:
        t: interval edges
        w: weights
        t_env: interval edges of the upper bound enveloping histogram (from proposal model)
        w_env: weights that should upper bound the inner (t,w) histogram (from proposal model)
    """
    # c: t, w, cp, wp
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        self.device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        if 'num_levels' in cfg.network: self.level = cfg.network.num_levels
        else: self.level = len(cfg.samplers)
        if cfg.loss.get('perc_loss', 0.) > 0: self.perc_loss = VGGPerceptualLoss().to(self.device)
    
    def compute_color_loss(self, output, batch, scalar_stats, suffix):
        rgb_map = output['rgb_map{}'.format(suffix)]
        if cfg.loss.get('rand_bkgd', False): assert(not cfg.white_bkgd); rgb_map += torch.rand_like(rgb_map) * (1. - output['acc_map{}'.format(suffix)][..., None])
        color_mse = self.color_crit(rgb_map, batch['rgb'])
        psnr = -10. * torch.log(color_mse.detach()) / math.log(10)
        scalar_stats.update({'psnr{}'.format(suffix): psnr})
        scalar_stats.update({'color_mse{}'.format(suffix): color_mse})
        return color_mse
    
    def compute_msk_loss(self, output, batch, scalar_stats, suffix):
        msk_mse = self.color_crit(output['acc_map{}'.format(suffix)], batch['msk'])
        scalar_stats.update({'msk_mse{}'.format(suffix): msk_mse})
        return cfg.loss.get('msk_weight') * msk_mse
        
    def compute_prop_loss(self, output, batch, scalar_stats, suffix):
        c = output['t_vals_0']
        w = output['weights_0']
        cp = output['t_vals{}'.format(suffix)]
        wp = output['weights{}'.format(suffix)]
        loss = torch.mean(lossfun_outer(c, w, cp, wp))
        scalar_stats.update({'prop_loss{}'.format(suffix): loss})
        return cfg.loss.get('prop_weight') * loss
    
    def compute_planetv_loss(self, output, batch, scalar_stats, suffix, level):
        model = output['model']
        if level == 0: encoder = model.xyz_encoder
        elif level == 1: encoder = model.sampler.network.xyz_encoder
        elif level == 2: encoder = model.sampler.network.sampler.network.xyz_encoder
        elif level == 3: encoder = model.sampler.network.sampler.network.sampler.network.xyz_encoder
        else: import ipdb; ipdb.set_trace()
        
        loss = 0. 
        spatial_embedding, temporal_embedding = encoder.spatial_embedding, encoder.temporal_embedding
        for i in range(encoder.num_res):
            loss += ( spatial_embedding[i][:, :, 1:, :] -  spatial_embedding[i][:, :, :-1, :]).pow(2).mean()
            loss += ( spatial_embedding[i][:, :, :, 1:] -  spatial_embedding[i][:, :, :, :-1]).pow(2).mean()
            loss += (temporal_embedding[i][:, :, 1:, :] - temporal_embedding[i][:, :, :-1, :]).pow(2).mean()
            loss += (temporal_embedding[i][:, :, :, 1:] - temporal_embedding[i][:, :, :, :-1]).pow(2).mean()
        scalar_stats.update({'plane_tv_loss{}'.format(suffix): loss})
        return cfg.loss.get('plane_tv') * loss
    
    def compute_timesmooth_loss(self, output, batch, scalar_stats, suffix, level):
        model = output['model']
        if level == 0: encoder = model.xyz_encoder
        elif level == 1: encoder = model.sampler.network.xyz_encoder
        elif level == 2: encoder = model.sampler.network.sampler.network.xyz_encoder
        elif level == 3: encoder = model.sampler.network.sampler.network.sampler.network.xyz_encoder
        else: import ipdb; ipdb.set_trace()
        
        loss = 0. 
        temporal_embedding = encoder.temporal_embedding
        for i in range(encoder.num_res):
            first_difference = (temporal_embedding[i][:, :, 1:, :] - temporal_embedding[i][:, :, :-1, :])
            second_difference = first_difference[:, :, 1:, :] - first_difference[:, :, :-1, :]
            loss += second_difference.pow(2).mean()
        scalar_stats.update({'time_smooth{}'.format(suffix): loss})
        return cfg.loss.get('time_smooth')[level] * loss
    
    def compute_distortion_loss(self, output, batch, scalar_stats, suffix):
        # TODO: only support Uniform samples: ENeRF, IBRNet
        weights = output['weights{}'.format(suffix)]
        t_vals = torch.linspace(0, 1., weights.shape[-1], device=weights.device, dtype=weights.dtype)[None, None, :].repeat(weights.shape[0], weights.shape[1], 1)
        t_matrix = t_vals[..., None] - t_vals[..., None, :]
        w_matrix = weights[..., None] * weights[..., None, :]
        loss_matrix = w_matrix * t_matrix.abs()
        loss = loss_matrix.sum(dim=-1).mean()
        scalar_stats.update({'distortion{}'.format(suffix): loss})
        return loss * cfg.loss.get('distortion')
    
    def compute_perc_loss(self, output, batch, scalar_stats, suffix):
        B = len(batch['rgb'])
        pred = output['rgb_map{}'.format(suffix)][:, cfg.num_pixels:].reshape(B*cfg.num_patches, cfg.patch_size, cfg.patch_size, 3).permute(0, 3, 1, 2)
        gt = batch['rgb'][:, cfg.num_pixels:].reshape(B*cfg.num_patches, cfg.patch_size, cfg.patch_size, 3).permute(0, 3, 1, 2) 
        loss = self.perc_loss(pred, gt)
        scalar_stats.update({'perc_loss{}'.format(suffix): loss})
        return loss * cfg.loss.get('perc_loss')
    
    def compute_loss_level(self, output, batch, scalar_stats, level=-1):
        suffix = '' if level == -1 else '_{}'.format(level)
        loss = 0 
        if 'rgb_map{}'.format(suffix) in output: loss += self.compute_color_loss(output, batch, scalar_stats, suffix) 
        if self.training and 'rgb_map{}'.format(suffix) in output and cfg.get('num_patches', 0) > 0 and cfg.loss.get('perc_loss', 0.) > 0.: loss += self.compute_perc_loss(output, batch, scalar_stats, suffix)
        if 'acc_map{}'.format(suffix) in output and 'msk' in batch and cfg.loss.get('msk_weight', 0.) > 0.: loss += self.compute_msk_loss(output, batch, scalar_stats, suffix) 
        if level >= 1 and cfg.loss.get('prop_weight', 0.) > 0. and 'weights{}'.format(suffix) in output: loss += self.compute_prop_loss(output, batch, scalar_stats, suffix) 
        if cfg.loss.get('plane_tv', 0.) > 0.: loss += self.compute_planetv_loss(output, batch, scalar_stats, suffix, level=level) 
        if cfg.loss.get('time_smooth', [0., 0., 0., 0., 0.])[level] > 0.: loss += self.compute_timesmooth_loss(output, batch, scalar_stats, suffix, level=level) 
        if cfg.loss.get('distortion', 0.) > 0. and 'weights{}'.format(suffix) in output: loss += self.compute_distortion_loss(output, batch, scalar_stats, suffix) 
        return loss

    def forward(self, batch):
        output = self.net(batch)
        loss = 0
        scalar_stats = {} 
        for i in range(self.level):
            loss += self.compute_loss_level(output, batch, scalar_stats, level=i) 
        scalar_stats.update({'loss': loss})
        image_stats = {}
        return output, loss, scalar_stats, image_stats