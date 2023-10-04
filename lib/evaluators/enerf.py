from lib.evaluators.nerf import Evaluator as BaseEvaluator
from lib.utils.data_utils import save_img, export_dir_to_video, save_depth, save_srcinps, dpt_convert_to_img, rgb_convert_to_img
from os.path import join
from lib.config import cfg

class Evaluator(BaseEvaluator):
    
    def save_results(self, output, batch, img_set):
        super().save_results(output, batch, img_set)
        if not cfg.save_enerf:
            return
        t_mean_std_min_max = output['batch_share_info']['t_mean_std_min_max'].permute(0, 2, 3, 1)
        meta = img_set['meta']
        ks = ['mean', 'std', 'min', 'max'] 
        idx = 0
        for k in ks:
            img_path = join(cfg.result_dir, 'step{:08d}/{}/{}'.format(self.step, k, meta))
            save_img(img_path, t_mean_std_min_max[0, :, :, idx].detach().cpu().numpy())
            idx += 1
        