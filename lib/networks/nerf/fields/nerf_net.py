import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from lib.networks.nerf.activations import get_activation

class NerfNet(nn.Module):
    def __init__(self,
                 xyz_in_dim: int,
                 view_in_dim: Optional[int] = None,
                 width: int = 256,
                 depth: int = 8,
                 view_depth: int = 1,
                 view_width: int = 128,
                 activation: str = 'relu',
                 skips: List[int] = [4],
                 **kwargs,
                 ):
        super(NerfNet, self).__init__()
        self.skips = skips
        self.backbone_nets = nn.ModuleList()
        segments = [0] + skips + [depth-1]
        for i in range(len(segments)-1):
            in_ch = xyz_in_dim if i == 0 else width + xyz_in_dim
            segment_net = nn.Sequential(*(
                [nn.Linear(in_ch, width), get_activation(activation)(inplace=True)] +
                [nn.Sequential(nn.Linear(width, width), get_activation(activation)(inplace=True)) for item in range(segments[i], segments[i+1])]
            ))
            self.backbone_nets.append(segment_net)
        self.sigma_net = nn.Linear(width, 1)
        self.rgb_net = nn.Sequential(*(
            [nn.Linear(view_in_dim + width, view_width), get_activation(activation)(inplace=True)] +
            [nn.Sequential(nn.Linear(view_width, view_width), get_activation(activation)(inplace=True)) for item in range(view_depth)] + 
            [nn.Linear(view_width, 3), nn.Sigmoid()]))

    def forward(self, xyz_encoding, view_encoding, only_geo=False):
        
        x = self.backbone_nets[0](xyz_encoding)
        for i, net in enumerate(self.backbone_nets[1:]):
            x = net(torch.cat([x, xyz_encoding], dim=-1))
        
        sigma = self.sigma_net(x)
        if only_geo: return None, sigma
        
        rgb = self.rgb_net(torch.cat([x, view_encoding], dim=-1))
        return rgb, sigma