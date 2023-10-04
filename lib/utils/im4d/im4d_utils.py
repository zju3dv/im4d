import torch
import numpy as np
import cv2
import open3d as o3d
import trimesh
class Im4DUtils:
    @staticmethod
    def render_median_depth(weights, z_vals):
        weights = weights[..., None]
        z_vals = z_vals[..., None]
        cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)
        split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
        median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
        median_index = torch.clamp(median_index, 0, z_vals.shape[-2] - 1)  # [..., 1]
        median_depth = torch.gather(z_vals[..., 0], dim=-1, index=median_index)  # [..., 1]
        return median_depth[..., 0]
    
    @staticmethod
    def extract_mesh(pts, msks, hws, rgbs, mask_at_boxs, export_path, erode_pixel=15):
        points, colors = [], []
        for pt, msk, hw, rgb, mask_at_box in zip(pts, msks, hws, rgbs, mask_at_boxs):
            h, w = hw
            if erode_pixel >= 2:
                errode_msk = torch.zeros((h, w), device=msk.device).float()
                errode_msk[mask_at_box.reshape(h, w)!=0] = msk
                errode_msk = (errode_msk > 0.5).detach().cpu().numpy().astype(np.uint8)
                errode_msk = errode_msk.reshape(h, w)
                kernel = np.ones((erode_pixel,erode_pixel),np.uint8)
                errode_msk = cv2.erode(errode_msk,kernel,iterations = 1)
                msk = errode_msk[mask_at_box.cpu().numpy().reshape(h, w)!=0]
            points.append(pt.detach().cpu().numpy()[msk.nonzero()])
            colors.append(rgb.detach().cpu().numpy()[msk.nonzero()])
        points = np.concatenate(points, axis=0)
        colors = (np.concatenate(colors, axis=0) * 255.).astype(np.uint8)
        mesh = trimesh.PointCloud(points, colors=colors)
        mesh.export(export_path)
        