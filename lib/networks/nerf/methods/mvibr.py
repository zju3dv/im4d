import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.config import cfg
from lib.networks.nerf.network import Network as NeRFNetwork
from lib.networks.nerf.methods.ibr.feature_nets import get_feature_net

class Network(NeRFNetwork):
    def __init__(self, net_cfg, sample_level=0):
        super(Network, self).__init__(net_cfg, sample_level=sample_level)
        if sample_level == 0: self.feature_net = get_feature_net(net_cfg.feature_net)
        
    def forward_feat(self, batch, batch_share_info):
        B, S, C, H, W = batch['src_inps'].shape
        img_feat = self.feature_net(batch['src_inps'].reshape(B*S, C, H, W)) # list of feature maps, sample_level
        img_feat = [item.reshape(B, S, item.shape[-3], item.shape[-2], item.shape[-1]) for item in img_feat]
        batch_share_info['img_feat'] = img_feat
        
    def forward(self, batch):
        batch_share_info = {}
        # Im4D Sec. 3.3
        if self.training and batch['step'] >= cfg.joint_iters and batch['step'] % cfg.finetune_iters_per != 0:
            with torch.no_grad():
                self.forward_feat(batch, batch_share_info)
        else: self.forward_feat(batch, batch_share_info)
        return super().forward(batch, batch_share_info)
