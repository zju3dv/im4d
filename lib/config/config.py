from .yacs import CfgNode as CN
import argparse
import os
from os.path import join
import numpy as np
from . import yacs
import yaml
import sys

import logging
import colorlog
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    log_colors={
        'DEBUG': 'yellow',   
        'INFO': 'white',    
        'WARNING': 'red',
        'ERROR': 'red,bg_white',     
        'CRITICAL': 'red,bg_white', 
    }
)
logger = logging.getLogger(__name__)


def define_basic_cfg():
    cfg = CN()
    cfg.machine = 'default'
    cfg.train_mode = sys.argv[0] == 'train_net.py'
    cfg.debug = False
    cfg.workspace = os.environ['workspace']
    logger.debug('workspace: ' + cfg.workspace)
    
    cfg.save_result = False
    cfg.clear_result = False
    cfg.save_tag = 'default'
    cfg.write_video = False
    cfg.depth_method = 'expected'
    # module
    cfg.train_dataset_module = 'lib.datasets.dtu.neus'
    cfg.test_dataset_module = 'lib.datasets.dtu.neus'
    cfg.network_module = 'lib.neworks.neus.neus'
    cfg.loss_module = 'lib.train.losses.neus'
    cfg.evaluator_module = 'lib.evaluators.neus'
    
    # experiment name
    cfg.exp_name = 'gitbranch_hello'
    cfg.exp_name_tag = ''
    cfg.grid_tag = 'default'
    cfg.pretrain = ''
    
    # training network
    cfg.distributed = False
    cfg.train_fp16 = False
    cfg.eval_fp16 = False
    
    # task
    cfg.task = 'hello'
    cfg.resume = True
    cfg.sigma_thresh = 5.
    cfg.fast_render = False
    
    # epoch
    cfg.ep_iter = -1
    cfg.save_ep = 1
    cfg.save_latest_ep = 1000
    cfg.eval_ep = 1
    cfg.log_interval = 1
    
    cfg.task_arg = CN()
    cfg.vr_weight_thresh = 0.0
    # -----------------------------------------------------------------------------
    # train
    # -----------------------------------------------------------------------------
    cfg.train = CN()
    cfg.train.epoch = 10000
    cfg.train.num_workers = 8
    cfg.train.collator = 'default'
    cfg.train.batch_sampler = 'default'
    cfg.train.sampler_meta = CN({})
    cfg.train.shuffle = True
    cfg.train.eps = 1e-8
    
    # use adam as default
    cfg.train.optim = 'adam'
    cfg.train.lr = 5e-4
    cfg.train.weight_decay = 0.
    cfg.train.scheduler = CN({'type': 'multi_step', 'milestones': [
                             80, 120, 200, 240], 'gamma': 0.5})
    cfg.train.batch_size = 4
    
    # test
    cfg.test = CN()
    cfg.test.batch_size = 1
    cfg.test.collator = 'default'
    cfg.test.epoch = -1
    cfg.test.num_workers = 0
    cfg.test.batch_sampler = 'default'
    cfg.test.sampler_meta = CN({})
    
    # trained model
    cfg.trained_model_dir = os.path.join(os.environ['workspace'], 'trained_model')
    cfg.clean_tag = 'debug'
    
    # recorder
    cfg.record_dir = os.path.join(os.environ['workspace'], 'record')
    
    # result
    cfg.result_dir = os.path.join(os.environ['workspace'], 'result')
    
    # evaluation
    cfg.skip_eval = False
    cfg.fix_random = False
    return cfg
    

def parse_cfg(cfg, args):
    if len(cfg.exp_name_tag) != 0: cfg.exp_name += ('_' + cfg.exp_name_tag)
    cfg.exp_name = cfg.exp_name.replace('FILENAME', os.path.basename(args.cfg_file).split('.')[0])
    cfg.exp_name = cfg.exp_name.replace('GITBRANCH', os.popen('git describe --all').readline().strip()[6:])
    cfg.exp_name = cfg.exp_name.replace('GITCOMMIT', os.popen('git describe --tags --always').readline().strip())
    cfg.exp_name = cfg.exp_name.replace('TODAY', os.popen('date +%Y-%m-%d').readline().strip())
    
    if cfg.get('separate', False):
        cfg.grid_dir = os.path.join(cfg.workspace, cfg.train_dataset.data_root, cfg.scene, 'grid', 'foreground')
        cfg.grid_dir_bg = os.path.join(cfg.workspace, cfg.train_dataset.data_root, cfg.scene, 'grid', 'background')
    else:
        cfg.grid_dir = os.path.join(cfg.result_dir, cfg.scene, cfg.task, 'grid', cfg.grid_tag)

    cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.scene, cfg.task, cfg.exp_name)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.scene, cfg.task, cfg.exp_name)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.scene, cfg.task, cfg.exp_name, cfg.save_tag)
    cfg.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0
    modules = [key for key in cfg if '_module' in key]
    for module in modules:
        logger.info(module + ' ' + cfg[module])
        cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'
        
def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def merge_cfg(cfg_file, cfg, loaded_cfg_files):
    with open(cfg_file, 'r') as f:
        current_cfg = yacs.load_cfg(f)
    if 'parent_cfg' in current_cfg.keys():
        cfg = merge_cfg(current_cfg.parent_cfg, cfg, loaded_cfg_files)
    if 'configs' in current_cfg.keys():
        for cfg_file_ in current_cfg.configs:
            cfg = merge_cfg(cfg_file_, cfg, loaded_cfg_files)
    cfg.merge_from_other_cfg(current_cfg)
    loaded_cfg_files.append(cfg_file)
    return cfg

def list_to_str(inputs, split='\n    '):
    ret = '['
    for item in inputs: ret += item + split
    return ret + ']'

def make_cfg(args):
    if args.cfg_file[:8] == 'configs/':
        cfg = define_basic_cfg()
        cfg.log_level = os.environ['LOG_LEVEL'] if ('LOG_LEVEL' in os.environ and os.environ['LOG_LEVEL'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) else args.log_level
        loaded_cfg_files = []
        cfg_ = merge_cfg(args.cfg_file, cfg, loaded_cfg_files)
        if args.configs is not None:
            for config in args.configs: merge_cfg(config, cfg_, loaded_cfg_files)
        logger.debug('Loaded config files (in order):')
        logger.debug(list_to_str(loaded_cfg_files))
        try: index = args.opts.index('other_opts'); cfg_.merge_from_list(args.opts[:index])
        except: cfg_.merge_from_list(args.opts)
        parse_cfg(cfg_, args)
        os.system('mkdir -p {}'.format(cfg_.result_dir))
    else:
        cfg = yaml.safe_load(open(args.cfg_file))
    if cfg.local_rank == 0:
        file_handler = logging.FileHandler(join(cfg_.result_dir, 'app.log'), mode='a' if cfg_.resume else 'w')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logger.addHandler(file_handler)
        file_handler.setFormatter(file_formatter)
    
    # LOGGING
    logger.info('=' * 30 + ' CONFIG ' + '=' * 30)
    logger.info('EXP NAME: ' + cfg.exp_name)
    if cfg.train_mode:
        logger.info('Training mode')
        logger.info('Training with float16: ' + str(cfg_.train_fp16))
        logger.info('Number of images: ' + str(cfg_.train_dataset.imgs_per_batch))
        logger.info('Number of pixels: ' + str(cfg_.num_pixels))
    else:
        logger.info('Testing mode')
        logger.info('Evaluation with float16: ' + str(cfg_.eval_fp16))
    logger.info('=' * 30 + ' CONFIG END ' + '=' * 30)
    return cfg_

def get_log_level(level_str):
    level_str = level_str.upper()
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return level_map.get(level_str, logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
parser.add_argument("--type", type=str, default="")
parser.add_argument("--configs", type=str, action='append')
parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the log level; It will firstly read LOG_LEVEL from environment variables', default='DEBUG')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
logger.setLevel(get_log_level(os.environ['LOG_LEVEL'] if ('LOG_LEVEL' in os.environ and os.environ['LOG_LEVEL'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']) else args.log_level))
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
cfg = make_cfg(args)