from typing import List
import torch
from kornia.utils import create_meshgrid
import torch.nn.functional as F
import numpy as np

class CostVolumeUtils:
    def homo_warp(src_feat, proj_mat, depth_values):
        B, D, H_T, W_T = depth_values.shape
        C, H_S, W_S = src_feat.shape[1:]
        device = src_feat.device

        R = proj_mat[:, :, :3] # (B, 3, 3)
        T = proj_mat[:, :, 3:] # (B, 3, 1)
        # create grid from the ref frame
        ref_grid = create_meshgrid(H_T, W_T, normalized_coordinates=False,
                               device=device) # (1, H, W, 2)
        ref_grid = ref_grid.permute(0, 3, 1, 2) # (1, 2, H, W)
        ref_grid = ref_grid.reshape(1, 2, H_T*W_T) # (1, 2, H*W)
        ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
        ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, 3, H*W)
        ref_grid_d = ref_grid.repeat(1, 1, D) # (B, 3, D*H*W)
        src_grid_d = R @ ref_grid_d + T/depth_values.view(B, 1, D*H_T*W_T)
        del ref_grid_d, ref_grid, proj_mat, R, T, depth_values # release (GPU) memory
        src_grid = src_grid_d[:, :2] / torch.clamp_min(src_grid_d[:, 2:], 1e-6) # divide by depth (B, 2, D*H*W)
        # del src_grid_d
        src_grid[:, 0] = (src_grid[:, 0])/((W_S - 1) / 2) - 1 # scale to -1~1
        src_grid[:, 1] = (src_grid[:, 1])/((H_S - 1) / 2) - 1 # scale to -1~1
        src_grid = src_grid.permute(0, 2, 1) # (B, D*H*W, 2)
        src_grid = src_grid.view(B, D, H_T*W_T, 2)
    
        warped_src_feat = F.grid_sample(src_feat, src_grid,
                                        mode='bilinear', padding_mode='border',
                                        align_corners=True) # (B, C, D, H*W)
        warped_src_feat = warped_src_feat.view(B, C, D, H_T, W_T)
        src_grid = src_grid.view(B, D, H_T, W_T, 2)
        if torch.isnan(warped_src_feat).isnan().any():
            __import__('ipdb').set_trace()
        return warped_src_feat, src_grid
    @staticmethod 
    def get_proj_mats(src_ext, src_ixt, tar_ext, tar_ixt, src_scale, tar_scale):
        B, S_V = src_ext.shape[:2]
        src_ixt = src_ixt.clone()
        src_ixt[:, :, :2] *= src_scale
        src_projs = src_ixt @ src_ext[:, :, :3]
    
        tar_ixt = tar_ixt.clone()
        tar_ixt[:, :2] *= tar_scale
        tar_projs = tar_ixt @ tar_ext[:, :3]
        tar_ones = torch.zeros((B, 1, 4)).to(tar_projs.device)
        tar_ones[:, :, 3] = 1
        tar_projs = torch.cat((tar_projs, tar_ones), dim=1)
        tar_projs_inv = torch.inverse(tar_projs)
    
        src_projs = src_projs.view(B, S_V, 3, 4)
        tar_projs_inv = tar_projs_inv.view(B, 1, 4, 4)
    
        proj_mats = src_projs @ tar_projs_inv
        return proj_mats
    
    @staticmethod
    def build_cost_volume(
        render_h_w: List[int], # [render_h, render_w]
        near_far: torch.Tensor, # Bx2
        tar_ext: torch.Tensor, # Bx4x4
        tar_ixt: torch.Tensor, # Bx3x3
        src_exts: torch.Tensor, # Bx4x4
        src_ixts: torch.Tensor, # Bx3x3
        img_feat: torch.Tensor, # BxSxCxHxW
        depth_inv: bool = False,
        num_planes: int = 64,
        tar_scale: float = 0.25,
        src_scale: float = 0.25,
        perturb: bool = False,
    ):
        B = len(near_far)
        h, w = render_h_w
        h, w = int(h * tar_scale), int(w * tar_scale)
        t_vals = torch.linspace(0., 1., steps=num_planes, device=near_far.device, dtype=torch.float32).view(1, -1, 1, 1).repeat(B, 1, h, w)
        if perturb:
            interval = 1/num_planes
            t_vals = np.random.random() * interval + t_vals 
        if depth_inv:
            t2z_func = lambda x, near_far: 1/(1/near_far[..., 0:1] + x*(1/near_far[..., 1:2] - 1/near_far[..., 0:1]))
        else:
            t2z_func = lambda x, near_far: near_far[..., 0:1] + x*(near_far[..., 1:2] - near_far[..., 0:1])
        B, D, H, W = t_vals.shape
        depth_values = t2z_func(t_vals.reshape(B, -1), near_far).reshape(B, D, H, W)
        proj_mats = CostVolumeUtils.get_proj_mats(src_ext=src_exts, 
                                                  src_ixt=src_ixts,
                                                  tar_ext=tar_ext,
                                                  tar_ixt=tar_ixt,
                                                  src_scale=src_scale, # ixt -> src_feat
                                                  tar_scale=tar_scale) # ixt -> depth_values)
        S = src_exts.shape[1]
        volume_sum = 0
        volume_sq_sum = 0
        for s in range(S):
            feature_s = img_feat[:, s]
            proj_mat = proj_mats[:, s]
            warped_volume, _ = CostVolumeUtils.homo_warp(feature_s, proj_mat, depth_values)
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        volume_variance = volume_sq_sum.div_(S).sub_(volume_sum.div_(S).pow_(2))
        return volume_variance, t_vals, t2z_func
        
    