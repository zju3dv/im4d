from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR
from lib.utils.kplanes.lr_scheduling import get_cosine_schedule_with_warmup
from lib.config import cfg as glo_cfg

def make_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs,
                                  gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'warmup_cosine':
        scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=128,
                num_training_steps=glo_cfg.train.epoch * glo_cfg.ep_iter
                )
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
    scheduler.gamma = cfg_scheduler.gamma
