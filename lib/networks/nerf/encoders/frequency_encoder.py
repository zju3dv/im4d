import torch
import torch.nn as nn
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FrequencyEncoder(nn.Module):
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
        in_dim: int, 
        num_frequencies: int, 
        include_input: bool = False,
        min_freq_exp: Optional[int] = None,
        max_freq_exp: Optional[int] = None,
        **kwargs
    ) -> None:
        super(FrequencyEncoder, self).__init__()
        self.in_dim = in_dim
        
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp or 0
        self.max_freq = max_freq_exp or self.num_frequencies - 1
        
        self.freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(device)
        
        self.include_input = include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim
    
    def forward(
        self, in_tensor, covs = None, **kwargs):
        
        scaled_inputs = in_tensor[..., None] * self.freqs        
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        # sinx cosx sin2x cos2x sin4x cos4x ...
        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
        else:
            input_var = torch.diagonal(covs, dim1=-2, dim2=-1)[..., :, None] * self.freqs[None, :] ** 2
            input_var = input_var.reshape((*input_var.shape[:-2], -1))
            encoded_inputs = expected_sin(
                torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1), torch.cat(2 * [input_var], dim=-1)
            )
        
        # reorder input to nerf order
        reorder_encoded_inputs = torch.zeros_like(encoded_inputs).view(*in_tensor.shape[:-1], -1, self.in_dim)
        encoded_inputs = encoded_inputs.view(*in_tensor.shape[:-1], 2, self.in_dim, self.num_frequencies).transpose(-2, -1) # [..., 2, "num_scales", "input_dim"]
        reorder_encoded_inputs[..., ::2, :] = encoded_inputs[..., 0, :, :]
        reorder_encoded_inputs[..., 1::2, :] = encoded_inputs[..., 1, :, :]
        encoded_inputs = reorder_encoded_inputs.reshape(*in_tensor.shape[:-1], -1) 
        
        if self.include_input:
            encoded_inputs = torch.cat([in_tensor, encoded_inputs], dim=-1)
        return encoded_inputs