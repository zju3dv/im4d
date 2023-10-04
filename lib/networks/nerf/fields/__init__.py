from lib.networks.nerf.fields.nerf_net import NerfNet
from lib.networks.nerf.fields.enerf_net import ENeRFNet
from lib.networks.nerf.fields.mini_ibr_net import MiniIBRNet
from lib.networks.nerf.fields.im4d_net import Im4DNet

nets_dict = {
    'nerf': NerfNet,
    'enerf': ENeRFNet,
    'mini_ibr': MiniIBRNet,
    'im4d': Im4DNet,
}
def get_net(net_cfg, **kwargs):
    return nets_dict[net_cfg.type](**kwargs)