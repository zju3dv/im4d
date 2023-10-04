import torch
from lib.utils.optimizer.radam import RAdam
from lib.config import logger


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def get_lr(lr_table, key):
    splits = key.split('.')
    for split in splits:
        if split in lr_table:
            return lr_table[split]
    return -1.

def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    lr_table = cfg.train.lr_table
    eps = cfg.train.eps

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        key_lr = get_lr(lr_table, key)
        if key_lr == 0.:
            logger.info('Skip learning for: ', key)
            continue
        elif key_lr > 0.:
            logger.info('Learning rate for {}: {}'.format(key, key_lr))
            params += [{"params": [value], "lr": key_lr, "weight_decay": weight_decay, "eps": eps}]
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "eps": eps}]
        # print('Learning rate for {}: {}'.format(key, lr))

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
