from lib.networks.nerf.fields.nerf_net import NerfNet
from lib.networks.nerf.fields.enerf_net import ENeRFNet

class Im4DNet(NerfNet):
    def __init__(self, **kwargs):
        super(Im4DNet, self).__init__(**kwargs)
        self.enerf_net = ENeRFNet(**kwargs)
        
    def forward(self, xyz_encoding, view_encoding, only_geo=False):
        _, sigma = super().forward(xyz_encoding[0], view_encoding, only_geo)
        if only_geo: return None, sigma
        # if self.training and kwargs['batch']['step'] >= cfg.joint_iters and kwargs['batch']['step'] % cfg.finetune_iters_per != 0:
        color, _ = self.enerf_net(xyz_encoding[1], view_encoding[..., :3])
        return color, sigma