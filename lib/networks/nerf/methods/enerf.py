import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.config import cfg
from lib.networks.nerf.methods.mvibr import Network as MvibrNetwork
from lib.networks.nerf.methods.ibr.cost_volume_utils import CostVolumeUtils
from lib.networks.nerf.methods.ibr.cost_reg_net import CostRegNet
from lib.utils.base_utils import get_xywh_tensor

class Network(MvibrNetwork):

    def __init__(self, net_cfg, sample_level=0):
        super(Network, self).__init__(net_cfg, sample_level=sample_level)
        self.depth_inv = net_cfg.depth_inv
        self.num_planes = net_cfg.num_planes
        self.mvs_scale = net_cfg.mvs_scale
        self.cost_feat_input_ch = net_cfg.cost_feat_input_ch
        self.cost_feat_level = net_cfg.cost_feat_level
        self.cost_feat_scale = net_cfg.cost_feat_scale
        self.cost_reg_net = CostRegNet(net_cfg.cost_feat_input_ch, light_weight=net_cfg.cost_net_light_weight)

    def compute_cost_volume(self, batch, batch_share_info):
        render_h, render_w = batch['meta']['H'][0].item(), batch['meta']['W'][0].item()
        render_h_w = [render_h, render_w]
        cost_volume, t_vals, t2z_func = CostVolumeUtils.build_cost_volume(
                render_h_w,
                batch['near_far'],
                batch['tar_ext'],
                batch['tar_ixt'],
                batch['src_exts'],
                batch['src_ixts'],
                batch_share_info['img_feat'][self.cost_feat_level],
                self.depth_inv,
                self.num_planes,
                perturb=self.training,
                tar_scale=self.mvs_scale,
                src_scale=self.cost_feat_scale)
        prob_volume = self.cost_reg_net(cost_volume)
        batch_share_info.update({
            't2z_func': t2z_func,
            't_vals': t_vals,
            'prob_volume': prob_volume
        })

    def predict_depth(self, batch, batch_share_info):
        prob_volume, t_vals = batch_share_info['prob_volume'], batch_share_info['t_vals']
        prob_volume = F.softmax(F.relu(prob_volume), dim=1)
        t_mean = (t_vals * prob_volume).sum(dim=1)
        t_std =  (prob_volume * (t_vals - t_mean.unsqueeze(1))**2).sum(1)
        t_std = t_std.clamp_min(1e-10).sqrt()
        t_min = torch.clamp(t_mean-t_std, min=0, max=None)
        t_max = torch.clamp(t_mean+t_std, min=None, max=1.)
        batch_share_info.update({'t_mean_std_min_max': torch.cat([t_mean[:, None], t_std[:, None], t_min[:, None], t_max[:, None]], dim=1)})

    def forward_feat(self, batch, batch_share_info):
        super().forward_feat(batch, batch_share_info)
        self.compute_cost_volume(batch, batch_share_info)
        self.predict_depth(batch, batch_share_info) # BxCxnum_planesxHxW, Bx1xnum_planesxHxW
