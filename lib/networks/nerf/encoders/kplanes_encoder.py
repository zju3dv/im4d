import torch
import torch.nn as nn
import torch.nn.functional as F


class KplanesEncoder(nn.Module):
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
        super().__init__()
        spatial_res, temporal_res = kwargs['spatial_res'], kwargs['temporal_res']
        assert(len(spatial_res) == len(temporal_res))
        num_feat_ch = kwargs['num_feat_ch']
        self.num_res, self.num_feat_ch, self.include_input = len(spatial_res), num_feat_ch, kwargs['include_input']
        self.spatial_embedding = nn.ParameterList([nn.Parameter(torch.zeros(3, num_feat_ch, spatial_res[i], spatial_res[i])) for i in range(self.num_res)])
        self.temporal_embedding = nn.ParameterList([nn.Parameter(torch.zeros(3, num_feat_ch, temporal_res[i], spatial_res[i])) for i in range(self.num_res)])
        
        self.res_composite = kwargs['res_composite']
        self.planes_composite = kwargs['planes_composite'] # concat, product, sum
        std = 1e-1
        for data in self.spatial_embedding: data.data.uniform_(-std, std)
        for data in self.temporal_embedding: data.data.uniform_(-std, std)
        if self.planes_composite == 'product':
            for data in self.spatial_embedding: nn.init.uniform_(data, a=0.1, b=0.15)
            for data in self.temporal_embedding: nn.init.ones_(data)
        
    def get_out_dim(self) -> int:
        return self.num_res * (self.num_feat_ch * 6 if self.planes_composite == 'concat' else self.num_feat_ch) + (4 if self.include_input else 0)
    
    def forward(
        self, in_tensor, **kwargs):
        # TODO: spatial
        sh = in_tensor.shape
        bbox = kwargs.get('bbox')
        xyz = (in_tensor.reshape(-1, 3) - bbox[:1]) / (bbox[1:] - bbox[:1])
        xyz = torch.clip(xyz, 0, 1)
        time = kwargs['time'].reshape(-1, 1)
        
        xy, yz, xz = xyz[:, :2], xyz[:, 1:], xyz[:, [0, 2]]
        spatial_coord = torch.stack([xy, yz, xz], dim=0) 
        xt, yt, zt = torch.cat([xyz[:, :1], time], dim=-1), torch.cat([xyz[:, 1:2], time], dim=-1), torch.cat([xyz[:, 2:], time], dim=-1)
        temporal_coord = torch.stack([xt, yt, zt], dim=0)
        
        output = []
        for i in range(self.num_res):
            spatial_embedding = F.grid_sample(self.spatial_embedding[i], spatial_coord[:, None]*2-1, align_corners=True, mode='bilinear')[:, :, 0].transpose(-1, -2)
            temporal_embedding = F.grid_sample(self.temporal_embedding[i], temporal_coord[:, None]*2-1, align_corners=True, mode='bilinear')[:, :, 0].transpose(-1, -2)
            embedding = torch.cat([spatial_embedding, temporal_embedding], dim=0)
            
            if self.planes_composite == 'concat': embedding = torch.cat([embedding[i] for i in range(len(embedding))], dim=-1)
            elif self.planes_composite == 'product': embedding = torch.prod(embedding, dim=0)
            elif self.planes_composite == 'sum': embedding = torch.sum(embedding, dim=0)
            else: import ipdb; ipdb.set_trace()
            
            output.append(embedding)
        if self.res_composite == 'concat': output = torch.cat(output, dim=-1)
        else: import ipdb; ipdb.set_trace()
        
        return output.reshape(sh[:-1] + (self.get_out_dim(),))