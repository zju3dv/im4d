from lib.networks.nerf.encoders.frequency_encoder import FrequencyEncoder
from lib.networks.nerf.encoders.none_encoder import NoneEncoder
from lib.networks.nerf.encoders.imgfeat_ibrnet import ImgfeatIbrnet
from lib.networks.nerf.encoders.tcnn_wrapper import TCNN
from lib.networks.nerf.encoders.kplanes_encoder import KplanesEncoder
from lib.networks.nerf.encoders.im4d_encoder import Im4DEncoder

encoding_dict = {
    'frequency': FrequencyEncoder,
    'none': NoneEncoder,
    'imgfeat_ibrnet': ImgfeatIbrnet,
    'tcnn': TCNN,
    'kplanes': KplanesEncoder,
    'im4d': Im4DEncoder,
}

def get_encoder(encoding_cfg):
    return encoding_dict[encoding_cfg.type](**encoding_cfg)
