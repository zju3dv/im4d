from lib.networks.nerf.fields.nerf_net import NerfNet
from lib.networks.nerf.fields.enerf_net import ENeRFNet
import torch

class Im4DNet(NerfNet):
    def __init__(self, **kwargs):
        super(Im4DNet, self).__init__(**kwargs)
        self.enerf_net = ENeRFNet(**kwargs)
        self.extra_color = kwargs.get('extra_color', False)
        # print('Extra color: ', self.extra_color)
        
    def forward(self, xyz_encoding, view_encoding, only_geo=False):
        color_, sigma = super().forward(xyz_encoding[0], view_encoding, only_geo)
        if only_geo: return None, sigma
        # if self.training and kwargs['batch']['step'] >= cfg.joint_iters and kwargs['batch']['step'] % cfg.finetune_iters_per != 0:
        color, _ = self.enerf_net(xyz_encoding[1], view_encoding[..., :3])
        # if self.training and self.extra_color: 
        #     color = torch.cat([color, color_], dim=-1)
        return color, sigma