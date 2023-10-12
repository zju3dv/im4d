import numpy as np
from lib.config import cfg
import cv2
from os.path import join
from lib.evaluators.base_evaluator import BaseEvaluator
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch

class Evaluator(BaseEvaluator):
    def __init__(self,):
        super(Evaluator, self).__init__()
        if 'lpips' in cfg.evaluate.metrics:
            import warnings
            # warnings.filterwarnings('ignore')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.loss_fn_vgg = lpips.LPIPS(net='vgg', verbose=False).cuda()
            # warnings.resetwarnings()
            # del warnings
            
    def parse_imgs_msk(self, output, batch, level=-1):
        suffix = '' if level == -1 else f'_{level}'
        h, w = batch['meta']['H'].item(), batch['meta']['W'].item()
        ret = {}
        mask_at_box = batch['mask_at_box'].detach().cpu().numpy().reshape(h, w)
        if level == -1: ret['eval_mask'] = mask_at_box 
        if level == -1: gt = np.ones((h, w, 3)) if cfg.white_bkgd else np.zeros((h, w, 3))
        rgb = np.ones((h, w, 3)) if cfg.white_bkgd else np.zeros((h, w, 3))
        depth = np.zeros((h, w))
        acc = np.zeros((h, w))
        if level == -1: gt[mask_at_box==1] = batch['rgb'].detach().cpu().numpy()[0]
        if f'rgb_map{suffix}' in output: 
            rgb[mask_at_box==1] = output[f'rgb_map{suffix}'].detach().cpu().numpy()[0]; 
            ret.update({f'rgb{suffix}': rgb.astype(np.float32)})
        if f'depth_map{suffix}' in output: 
            depth[mask_at_box==1] = output[f'depth_map{suffix}'].detach().cpu().numpy()[0]; 
            ret.update({f'depth{suffix}': depth.astype(np.float32)})
        if f'acc_map{suffix}' in output: 
            acc[mask_at_box==1] = output[f'acc_map{suffix}'].detach().cpu().numpy()[0]; 
            ret.update({f'acc{suffix}': acc.astype(np.float32)})
        if level == -1: 
            ret['gt'] = gt.astype(np.float32);
            view_id = batch['meta']['view_id'].item()
            view_id = batch['meta']['idx'].item() if view_id == -1 else view_id
            ret['meta'] = 'frame{:04d}_view{:04d}.png'.format(batch['meta']['frame_id'].item(), view_id) 
        return ret
        
    def parse_imgs(self, output, batch, level=-1):
        
        if 'mask_at_box' in batch:
            ret = {}
            ret.update(self.parse_imgs_msk(output, batch))
            for i in range(self.num_levels):
                ret.update(self.parse_imgs_msk(output, batch, level=i))
            return ret
        
        ret = {}
        h, w = batch['meta']['H'].item(), batch['meta']['W'].item()
        if level == -1:
            gt = batch['rgb'].detach().cpu().numpy().reshape(h, w, 3)
            view_id = batch['meta']['view_id'].item()
            view_id = batch['meta']['idx'].item() if view_id == -1 else view_id
            ret['meta'] = 'frame{:04d}_view{:04d}.png'.format(batch['meta']['frame_id'].item(), view_id)
            ret['eval_mask'] = np.ones_like(gt[..., 0]).astype(np.uint8)
            ret['gt'] = gt.astype(np.float32)
            for i in range(self.num_levels):
                ret.update(self.parse_imgs(output, batch, level=i))
        
        suffix = '' if level == -1 else f'_{level}'
        
        if f'rgb_map{suffix}' in output:  ret.update({f'rgb{suffix}': output[f'rgb_map{suffix}'].detach().cpu().numpy().reshape(h, w, 3).astype(np.float32)})
        if f'depth_map{suffix}' in output: ret.update({f'depth{suffix}': output[f'depth_map{suffix}'].detach().cpu().numpy().reshape(h, w).astype(np.float32)})
        if f'acc_map{suffix}' in output:   ret.update({f'acc{suffix}': output[f'acc_map{suffix}'].detach().cpu().numpy().reshape(h, w).astype(np.float32)})

        return ret
    
    def metric_psnr(self, output, batch, img_set, level=-1):
        suffix = '' if level == -1 else f'_{level}'
        if f'rgb{suffix}' not in img_set: return None
        psnr_item = psnr(img_set['gt'][img_set['eval_mask']==1], img_set[f'rgb{suffix}'][img_set['eval_mask']==1], data_range=1.)
        return psnr_item
        
    def metric_ssim(self, output, batch, img_set, level=-1):
        suffix = '' if level == -1 else f'_{level}'
        if f'rgb{suffix}' not in img_set: return None
        x, y, w, h = cv2.boundingRect(img_set['eval_mask'])
        ssim_item = ssim(img_set['gt'][y:y+h, x:x+w], img_set[f'rgb{suffix}'][y:y+h, x:x+w], data_range=2., channel_axis=-1)
        return ssim_item
        
    def metric_lpips(self, output, batch, img_set, level=-1):
        suffix = '' if level == -1 else f'_{level}'
        if f'rgb{suffix}' not in img_set: return None
        x, y, w, h = cv2.boundingRect(img_set['eval_mask'])
        gt, pred = img_set['gt'][y:y+h, x:x+w], img_set[f'rgb{suffix}'][y:y+h, x:x+w]
        gt, pred = torch.from_numpy(gt).cuda(), torch.from_numpy(pred).cuda()
        gt, pred = gt.permute(2, 0, 1).unsqueeze(0), pred.permute(2, 0, 1).unsqueeze(0)
        lpips_item = self.loss_fn_vgg(gt * 2. -  1., pred * 2 - 1.).item()
        return lpips_item
