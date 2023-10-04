import torch
import torch.nn as nn
try:
    import tinycudann as tcnn
except:
    pass

class TCNN(nn.Module):
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
        del kwargs['type']
        self.in_dim = kwargs['in_dim']
        self.encoder = tcnn.Encoding(kwargs['in_dim'], kwargs)
        self.include_input = kwargs['include_input']
        
    def get_out_dim(self) -> int:
        return self.encoder.n_output_dims + self.in_dim if self.include_input else self.encoder.n_output_dims
    
    def forward(
        self, in_tensor, **kwargs):
        # TODO: spatial
        sh = in_tensor.shape
        bbox = kwargs.get('bbox')
        xyz = (in_tensor.reshape(-1, 3) - bbox[:1]) / (bbox[1:] - bbox[:1])
        xyz = torch.clip(xyz, 0, 1)
        xyz_embedding = self.encoder(xyz)
        if self.include_input: xyz_embedding = torch.cat([xyz, xyz_embedding], dim=-1)
        return xyz_embedding.reshape(sh[:-1] + (self.get_out_dim(),))
        