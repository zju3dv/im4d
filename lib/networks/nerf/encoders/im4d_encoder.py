import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

from lib.networks.nerf.encoders.imgfeat_ibrnet import ImgfeatIbrnet
from lib.networks.nerf.encoders.kplanes_encoder import KplanesEncoder


class Im4DEncoder(KplanesEncoder):
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
        **kwargs
    ) -> None:
        super(Im4DEncoder, self).__init__(**kwargs)
        self.feature_projection = ImgfeatIbrnet(**kwargs)
        
    
    def forward(
        self, in_tensor, only_geo=False, **kwargs):
        geo_feat = super().forward(in_tensor, **kwargs)
        if only_geo: return geo_feat
        # if self.training and kwargs['batch']['step'] >= cfg.joint_iters and kwargs['batch']['step'] % cfg.finetune_iters_per != 0:
            # with torch.no_grad():
                # ibr_feat = self.feature_projection(in_tensor, **kwargs)
        # else: ibr_feat = self.feature_projection(in_tensor, **kwargs)
        ibr_feat = self.feature_projection(in_tensor, **kwargs)
        return (geo_feat, ibr_feat)