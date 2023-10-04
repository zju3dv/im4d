import torch
import torch.nn as nn


class NoneEncoder(nn.Module):
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
        super(NoneEncoder, self).__init__()
        
    def get_out_dim(self) -> int:
        return 0
    
    def forward(
        self, in_tensor, covs = None):
        return None
        