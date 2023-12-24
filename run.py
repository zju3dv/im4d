from lib.config import cfg, args
import numpy as np
import os
import cv2
from os.path import join
import torch
torch.backends.cudnn.benchmark = False

def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm
    
    from lib.utils.data_utils import save_img

    # cfg.train.num_workers = 8
    data_loader = make_data_loader(cfg, is_train=True)
    for batch in tqdm.tqdm(data_loader):
        print(batch['near_far'])

def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils.data_utils import to_cuda
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        batch = to_cuda(batch)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            with torch.cuda.amp.autocast(enabled=cfg.eval_fp16):
                network(batch)
            torch.cuda.synchronize()
            net_time.append(time.time() - start)
    if len(net_time) > 1:
        print('net_time: ', np.mean(net_time[1:]))
        print('fps: ', 1./np.mean(net_time[1:]))
    else:
        print('net_time: ', np.mean(net_time))
        print('fps: ', 1./np.mean(net_time))

def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    import time
    from lib.utils.net_utils import save_trained_config as save_config

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()
    save_config(cfg, 'test')

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.cuda.amp.autocast(enabled=cfg.eval_fp16):
                output = network(batch)
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
    evaluator.summarize()
    if len(net_time) > 1:
        print('net_time: ', np.mean(net_time[1:]))
        print('fps: ', 1./np.mean(net_time[1:]))
    else:
        print('net_time: ', np.mean(net_time))
        print('fps: ', 1./np.mean(net_time))

def run_cache_grid():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    import time
    from lib.utils.net_utils import save_trained_config as save_config

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()
    save_config(cfg, 'test')

    data_loader = make_data_loader(cfg, is_train=False)
    binarys = []
    bounds = []
    net_time = []
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.cuda.amp.autocast(enabled=cfg.eval_fp16):
                binary, bound = network.cache_grid(batch, cfg.save_mesh)
                binarys.append(binary)
                bounds.append(bound)
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
    os.makedirs(cfg.grid_dir, exist_ok=True)
    np.savez_compressed(join(cfg.grid_dir, 'binarys.npz'), np.array(binarys))
    np.savez_compressed(join(cfg.grid_dir, 'bounds.npz'), np.array(bounds))
    if len(net_time) > 1:
        print('net_time: ', np.mean(net_time[1:]))
        print('fps: ', 1./np.mean(net_time[1:]))
    else:
        print('net_time: ', np.mean(net_time))
        print('fps: ', 1./np.mean(net_time))

def run_export_pcd():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    import time
    from lib.utils.net_utils import save_trained_config as save_config
    from lib.utils.im4d.im4d_utils import Im4DUtils

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()
    save_config(cfg, 'test')

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    net_time = []
    frame_id = None
    pts, msks, hws, rgbs, mask_at_boxs = [], [], [], [], []
    for batch in tqdm.tqdm(data_loader):
        if frame_id is not None and batch['meta']['frame_id'].item() != frame_id:
            export_path = join(cfg.result_dir, 'meshes', '{:06d}.ply'.format(frame_id))
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            Im4DUtils.extract_mesh(pts, msks, hws, rgbs, mask_at_boxs, export_path)
            pts, msks, hws, rgbs, mask_at_boxs = [], [], [], [], []
        frame_id = batch['meta']['frame_id'].item()
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.cuda.amp.autocast(enabled=cfg.eval_fp16):
                output = network(batch)
            pts.append(output['pts_0'][0])
            msks.append(batch['msk'][0])
            rgbs.append(batch['rgb'][0])
            hws.append((batch['meta']['H'].item(), batch['meta']['W'].item()))
            mask_at_boxs.append(batch['mask_at_box'][0])
            torch.cuda.synchronize()
            end_time = time.time()
        net_time.append(end_time - start_time)
        evaluator.evaluate(output, batch)
    export_path = join(cfg.result_dir, 'meshes', '{:06d}.ply'.format(frame_id))
    Im4DUtils.extract_mesh(pts, msks, hws, rgbs, mask_at_boxs, export_path)
    evaluator.summarize()
    if len(net_time) > 1:
        print('net_time: ', np.mean(net_time[1:]))
        print('fps: ', 1./np.mean(net_time[1:]))
    else:
        print('net_time: ', np.mean(net_time))
        print('fps: ', 1./np.mean(net_time))

if __name__ == '__main__':
    globals()['run_' + args.type]()
