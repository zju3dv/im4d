import torch

class FastRendUtils:
    @staticmethod
    def prepare_wpts(wbounds, grid_res, subgrid_res):
        grid_res = torch.tensor(grid_res, device=wbounds.device, dtype=torch.float32)
        globol_domain_min, global_domain_max = wbounds[0], wbounds[1]
        global_domain_size = global_domain_max - globol_domain_min
        voxel_size = global_domain_size / grid_res
        voxel_offset_min = globol_domain_min
        voxel_offset_max = voxel_offset_min + voxel_size
        voxel_samples = []
        for dim in range(3): # TODO: for subgrid resolution
            voxel_samples.append(torch.linspace(voxel_offset_min[dim], voxel_offset_max[dim], subgrid_res[dim], device=wbounds.device))
        voxel_samples = torch.stack(torch.meshgrid(*voxel_samples, indexing='ij'), axis=-1).reshape(-1, 3)
        voxel_samples = voxel_samples[None, None, None] # [1, 1, 1, r, 3]
        voxel_ranges = []
        for dim in range(3):
            voxel_ranges.append(torch.arange(0, int(grid_res[dim].item()), device=wbounds.device))    
        voxel_grid = torch.stack(torch.meshgrid(*voxel_ranges, indexing='ij'), axis=-1)
        voxel_grid = (voxel_grid * voxel_size)[..., None, :] # [r1, r2, r3, -1, 3]
        pts = voxel_grid + voxel_samples # [r1, r2, r3, r, 3]
        return pts
