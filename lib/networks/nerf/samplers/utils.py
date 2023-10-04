import torch

def pdf_sample(spacing_vals, weights, training, num_samples, hist_padding=1e-2, eps=1e-5):
    num_bins = num_samples + 1

    weights = weights + hist_padding
    weights_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.relu(eps - weights_sum)
    weights = weights + padding / weights.shape[-1]
    weights_sum += padding

    pdf = weights / weights_sum
    cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
    u = u.expand((*cdf.shape[:-1], num_bins))
    if training:
        rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
        u = u + rand
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    existing_bins = spacing_vals
    below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
    above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
    cdf_g0 = torch.gather(cdf, -1, below)
    bins_g0 = torch.gather(existing_bins, -1, below)
    cdf_g1 = torch.gather(cdf, -1, above)
    bins_g1 = torch.gather(existing_bins, -1, above)
    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    bins = bins_g0 + t * (bins_g1 - bins_g0)
    return bins
