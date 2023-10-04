import numpy as np
from lib.config import cfg, logger
import os
import json
from os.path import join
from lib.utils.data_utils import save_img, export_dir_to_video, save_depth, save_srcinps, dpt_convert_to_img, rgb_convert_to_img, export_depth_to_pcd
from termcolor import colored

class BaseEvaluator:
    def __init__(self,):
        self.step = 999999
        if cfg.clear_result: os.system('rm -rf ' + join(cfg.result_dir, 'step{:08d}'.format(self.step), '*'))
        os.system('mkdir -p ' + cfg.result_dir)
        # meta information
        if 'num_levels' in cfg.network:
            self.num_levels = cfg.network.num_levels
        else:
            self.num_levels = len(cfg.samplers)
        self.reset_metrics()
        self.evaluate_metrics = cfg.evaluate.metrics
        
    def reset_metrics(self,):
        self.metrics = {}
        for level in range(self.num_levels):
            self.metrics.update({k+f'_{level}':[] for k in cfg.evaluate.metrics})
        
    def update_meta(self, output, batch):
        if 'step' in batch:
            self.step = batch['step']
            
    def update_metrics(self, output, batch, img_set):
        metric_stats = img_set['meta'] + ' '
        for level in range(self.num_levels):
            for k in self.evaluate_metrics:
                metric_item = getattr(self, 'metric_' + k)(output, batch, img_set, level=level)
                if metric_item is not None: self.metrics[k + f'_{level}'].append(metric_item)
                if len(self.metrics[k + f'_{level}']) == 0: continue
                metric_stats += '{}: {:.3f} '.format(k + f'_{level}', self.metrics[k + f'_{level}'][-1])
        logger.info(metric_stats)
    
    def save_results(self, output, batch, img_set):
        if not cfg.save_result:
            return
        meta = img_set['meta']
        for k in img_set:
            for kk in cfg.evaluate.save_keys:
                if kk in k:
                    acc_k = k.replace(kk, 'acc')
                    if cfg.evaluate.get('save_with_alpha', True) and acc_k in img_set: acc = img_set[acc_k]
                    else: acc = None
                    img_path = join(cfg.result_dir, 'step{:08d}/{}/{}'.format(self.step, k, meta))
                    if 'depth' in k:
                        save_depth(img_path, img_set[k], acc_map=acc)
                    else:
                        save_img(img_path, img_set[k], acc_map=acc)
        if 'src_inps' in cfg.evaluate.save_keys:
            img_path = join(cfg.result_dir, 'step{:08d}/{}/{}'.format(self.step, 'src_inps', meta))
            save_srcinps(img_path, batch['src_inps'].detach().cpu().numpy()[0])
            
    def parse_imgs(self, output, batch):
        pass
        
    def evaluate(self, output, batch):
        self.update_meta(output, batch)
        img_set = self.parse_imgs(output, batch)
        # export_depth_to_pcd(img_set['depth_0'] / np.clip(img_set['acc_0'], 0.01, 1.), batch['tar_ixt'][0].detach().cpu().numpy(), 'debug_depth0.ply')
        # export_depth_to_pcd(img_set['depth_1'] / np.clip(img_set['acc_1'], 0.01, 1.), batch['tar_ixt'][0].detach().cpu().numpy(), 'debug_depth1.ply')
        self.update_metrics(output, batch, img_set)
        self.save_results(output, batch, img_set)
        return self.to_image_stats(img_set)
    
    def to_image_stats(self, img_set):
        image_stats = {}
        
        for k in img_set:
            for kk in cfg.evaluate.save_keys:
                if kk in k:
                    acc_k = k.replace(kk, 'acc')
                    if cfg.evaluate.get('save_with_alpha', True) and acc_k in img_set: acc = img_set[acc_k]
                    else: acc = None
                    if 'depth' in k:
                        image_stats[k + '_' + img_set['meta']] = dpt_convert_to_img(img_set[k], acc).transpose(2, 0, 1)
                    elif 'rgb' in k:
                        image_stats[k + '_' + img_set['meta']] = rgb_convert_to_img(img_set[k], acc).transpose(2, 0, 1)
                    else:
                        pass
        return image_stats
        
    def ret_metrics(self,):
        ret = {k:float(np.mean(self.metrics[k])) for k in self.metrics if len(self.metrics[k]) > 0}
        self.reset_metrics()
        print('\n', '=' * 80, '\n', ret, '\n', '=' * 80)
        if cfg.save_result: print(colored('Results saved to {}'.format(cfg.result_dir), color='green'))
        os.makedirs(join(cfg.result_dir, 'step{:08d}'.format(self.step)), exist_ok=True)
        json.dump(ret, open(os.path.join(cfg.result_dir, 'step{:08d}'.format(self.step), 'metrics.json'), 'w'))
        return ret
        
    def write_videos(self,):
        if not cfg.get('write_video', False):
            return
        save_dir = join(cfg.result_dir, 'step{:08d}'.format(self.step))
        # 获取save_dir下所有目录
        dirs = [item for item in os.scandir(save_dir) if item.is_dir()]
        for dir in dirs:
            for k in cfg.evaluate.save_keys:
                if k in dir.name:
                    image_dir = dir.path
                    video_path = join(save_dir, 'mp4s/{}.mp4'.format(dir.name))
                    os.system('mkdir -p {}'.format(os.path.dirname(video_path)))
                    export_dir_to_video(image_dir, video_path, fps=cfg.fps)
        print(colored('Videos saved to {}'.format(join(save_dir, 'mp4s')), color='green'))
            
    def summarize(self):
        # eval and save metrics
        ret = self.ret_metrics()
        self.write_videos()
        return ret
