from lib.networks.nerf.methods.ibr.feature_nets.featurenet_unet import FeatureNet as ENeRFUnet

nets_dict = {
    'enerf_unet': ENeRFUnet
}
def get_feature_net(net_cfg, **kwargs):
    return nets_dict[net_cfg.type](**net_cfg)