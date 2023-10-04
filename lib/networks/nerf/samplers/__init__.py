from lib.networks.nerf.samplers.cascade_sampler import CascadeSampler
from lib.networks.nerf.samplers.uniform_sampler import UniformSampler
from lib.networks.nerf.samplers.dispuniform_sampler import DispUniformSampler
from lib.networks.nerf.samplers.mvs_sampler import MvsSampler

samplers_dict = {
    'mvs_sampler': MvsSampler,
    'cascade_sampler': CascadeSampler,
    'uniform_sampler': UniformSampler,
    'dispuniform_sampler': DispUniformSampler,
}

def get_sampler(sampler_cfg, global_cfg=None, sample_level=0):
    return samplers_dict[sampler_cfg['type']](global_cfg=global_cfg, sample_level=sample_level+1, **sampler_cfg)