import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from lib.utils.im4d.im4d_utils import Im4DUtils

class ImgfeatIbrnet(nn.Module):
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
        feat_scale = 0.25,
        feat_dim = 32,
        **kwargs
    ) -> None:
        super(ImgfeatIbrnet, self).__init__()
        self.feat_dim = feat_dim
        self.feat_scale = feat_scale

    def get_out_dim(self) -> int:
        return 3 + self.feat_dim + 4 + 1
    
    def forward(
        self, xyz, viewdir=None, **kwargs):
        # xyz: BxN_raysxN_samplesx3
        if cfg.get('ndc', False): xyz = Im4DUtils.ndc2world(xyz, cfg.ndc_FOCAL, cfg.ndc_H, cfg.ndc_W)
        sample_level, batch = kwargs['sample_level'], kwargs['batch']
        if xyz.shape[-1] == 3: viewdir = viewdir if viewdir is not None else kwargs['viewdir']
        else: xyz, viewdir = torch.split(xyz, [3, 3], dim=-1)
        img_feat = kwargs['batch_share_info']['img_feat'][sample_level]
        src_inps, src_exts, src_ixts = batch['src_inps'], batch['src_exts'], batch['src_ixts']
        B, S, C, H, W = src_inps.shape
        src_exts = src_exts.reshape(B * S, 4, 4)
        src_ixts = src_ixts.reshape(B * S, 3, 3)
        src_inps = src_inps.reshape(B * S, C, H, W) * 0.5 + 0.5
        F_C, F_H, F_W = img_feat.shape[2:]
        src_feat = img_feat.reshape(B*S, F_C, F_H, F_W)
        
        N_rays, N_samples = xyz.shape[1], xyz.shape[2]
        view_xyz = xyz[:, None].repeat(1, S, 1, 1, 1).reshape(B * S, N_rays * N_samples, 3) # B*S x N_rays*N_samples x 3
        if len(viewdir.shape) == 4: viewdir = viewdir[:, None].repeat(1, S, 1, 1, 1).reshape(B * S, N_rays * N_samples, 3)
        else: viewdir = viewdir[:, None, :, None].repeat(1, S, 1, N_samples, 1).reshape(B * S, N_rays * N_samples, 3)# B*S x N_rays*N_samples x 3
        
        point_src_diff = src_exts.inverse()[:, :3, 3:].transpose(-1, -2) - view_xyz
        point_src_diff = point_src_diff / (point_src_diff.norm(dim=-1, keepdim=True) + 1e-6)
        point_tar_diff = -viewdir
        point_diff_dot = (point_tar_diff * point_src_diff).sum(dim=-1, keepdim=True)
        point_diff = point_tar_diff - point_src_diff
        point_diff = point_diff / (point_diff.norm(dim=-1, keepdim=True) + 1e-6)
        point_dir = torch.cat([point_diff, point_diff_dot], dim=-1)
        
        view_xyz = view_xyz @ src_exts[:, :3, :3].transpose(-1, -2) + src_exts[:, :3, 3:].transpose(-1, -2)
        view_xyz = view_xyz @ src_ixts[:, :3, :3].transpose(-1, -2)
        
        # view_xyz_ = view_xyz[..., :2] / torch.clamp_min(view_xyz[..., 2:], 1e-6)
        # view_xyz_[..., 0] /= (W-1)
        # view_xyz_[..., 1] /= (H-1)
        # xyz_depth = view_xyz
        # view_xyz = view_xyz_
        grid_xyz = view_xyz[..., :2] / torch.clamp_min(view_xyz[..., 2:], 1e-6)
        grid_xyz[..., 0] /= (W-1)
        grid_xyz[..., 1] /= (H-1)
        
        point_msk = ((grid_xyz[..., 0] >= 0) & (grid_xyz[..., 0] <= 1) & (grid_xyz[..., 1] >= 0) & (grid_xyz[..., 1] <= 1) & (view_xyz[..., 2] >= 1e-6)).float()
        point_rgb = F.grid_sample(src_inps, grid_xyz[..., None, :2]*2-1, align_corners=True, padding_mode='border')[..., 0].transpose(-1, -2)
        point_feat = F.grid_sample(src_feat, grid_xyz[..., None, :2]*2-1, align_corners=True, padding_mode='border')[..., 0].transpose(-1, -2)
        
        point_feat = torch.cat([point_rgb, point_feat, point_dir, point_msk[..., None]], dim=-1)
        
        return point_feat.reshape(B, S, N_rays, N_samples, self.get_out_dim()) # BxSxN_raysxN_samplesx(out_dim)