# Im4D: High-Fidelity and Real-Time Novel View Synthesis for Dynamic Scenes

### [Project Page](https://zju3dv.github.io/im4d) | [Paper](https://drive.google.com/file/d/1MOixYy-TESDvcoL9Qj4V7tDvafqDmibh/view?usp=sharing) | [Video](https://youtu.be/pPl1M5jpK4g)
> [High-Fidelity and Real-Time Novel View Synthesis for Dynamic Scenes](https://drive.google.com/file/d/1MOixYy-TESDvcoL9Qj4V7tDvafqDmibh/view?usp=sharing) \
> Haotong Lin, Sida Peng, Zhen Xu, Tao Xie, Xingyi He, Hujun Bao and Xiaowei Zhou \
> SIGGRAPH Asia 2023 conference track

![DNA-Rendering](https://github.com/haotongl/imgbed/raw/master/im4d/renbody.gif)

<!-- ![ENeRF-Outdoor](https://github.com/haotongl/imgbed/raw/master/im4d/enerf.gif) -->

## Installation

### Set up the python environment
<details> <summary>Tested with an Ubuntu workstation i9-12900K, 3090GPU</summary>

```
conda create -n im4d python=3.10
conda activate im4d
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # pytorch 2.0.1
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
pip install -r requirments.txt
```
</details>

### Set up datasets

<details> <summary>0. Set up workspace</summary>

The workspace is the disk directory that stores datasets, training logs, checkpoints and results. Please ensure it has enough disk space. 

```
export workspace=$PATH_TO_YOUR_WORKSPACE
```
</details>

<details> <summary>1. Prepare ZJU-MoCap and NHR datasets.</summary>

Please refer to [mlp_maps](https://github.com/zju3dv/mlp_maps/blob/master/INSTALL.md) to download ZJU-MoCap and NHR datasets.
After downloading, place them into `$workspace/zju-mocap` and `$workspace/NHR`, respectively.
</details>
<details> <summary>2. Prepare the DNA-Rendering dataset.</summary>

Since the license of the [DNA-Rendering](https://dna-rendering.github.io/index.html) dataset does not allow distribution, we cannot release the processed dataset publicly.
You can download the DNA-Rendering dataset at [here](https://dna-rendering.github.io/inner-download.html) or [OpenXLab](https://openxlab.org.cn/datasets/OpenXDLab/DNA-Rendering) . If someone is interested at the processed data, please email me (haotongl@outlook.com).
You need to cite [DNA-Rendering](https://dna-rendering.github.io/index.html) if you find this data useful.
</details> 

<!-- <details> <summary>3. [TODO] Prepare the dynerf dataset.</summary> -->
<!-- </details> -->

<!-- <details> <summary>4. [TODO] Prepare the ENeRF-Outdoor dataset.</summary> -->
<!-- </details> -->

### Pre-trained models

Download pre-trained models from [this link](https://drive.google.com/drive/folders/1_huSP1XOG-HttZwu-JxmICrsR9YQOpkm?usp=sharing) for quick test. Place FILENAME.pth into\
`$workspace/trained_model/SCENE/im4d/FILENAME/latest.pth`. \
e.g., my_313.pth -> `$workspace/trained_model/my_313/im4d/my_313/latest.pth` \
my_313_demo.pth -> `$workspace/trained_model/my_313/im4d/my_313_demo/latest.pth`.

## Testing

<details> <summary>1. Reproduce the quantitative results in the paper.</summary>

```
python run.py --type evaluate --cfg_file configs/exps/im4d/xx_dataset/xx_scene.yaml save_result True
```

For the NHR dataset, please firstly download [the preprocessed data](https://drive.google.com/drive/folders/1rA1gzzub6TkGIuu-LaqYwwwiJm4svK2F?usp=sharing) and place them into `$workspace/evaluation`. This evaluation setting is taken from [mlp_maps](https://zju3dv.github.io/mlp_maps/).
Then run one more command to report the PSNR metric:
```
python scripts/evaluate/im4d/eval_nhr.py --gt_path $workspace/evaluation/sport_1_easymocap --output_path $workspace/result/sport_1_easymocap/im4d/sport1_release/default/step00999999/rgb_0
```
</details>

<details> <summary>2. Accelerate the rendering speed .</summary>
First, precompute the binary fields.

```
python run.py --type cache_grid --cfg_file configs/exps/im4d/renbody/0013_01.yaml --configs configs/components/opts/cache_grid.yaml grid_tag default
```
You may need to change the frames and grid_resolution to fit your scene. 
For example, the scene in ZJU-MoCap has 300 frames and its height is z-axis:
```
python run.py --type cache_grid --cfg_file configs/exps/im4d/zju/my_313.yaml --configs configs/components/opts/cache_grid.yaml grid_tag default grid_resolution 128,128,256 test_dataset.frame_sample 0,300,1
```


Then, render images with the precomputed binary fields.

```
python run.py --type evaluate --cfg_file configs/exps/im4d/renbody/0013_01.yaml --configs configs/components/opts/fast_render.yaml grid_tag default save_result True
```

You may try slightly decreasing sigma_thresh (default: 5.0) to preserve more voxels.

</details>


<details> <summary>3. Render a video with the selected trajectory.</summary>


```
python run.py --type evaluate --cfg_file configs/exps/im4d/renbody/0013_01.yaml --configs configs/components/opts/render_path/renbody_path.yaml
```
We can render it with the precomputed binary fields by adding one more argument:

```
python run.py --type evaluate --cfg_file configs/exps/im4d/renbody/0013_01.yaml --configs configs/components/opts/render_path/renbody_path.yaml --configs configs/components/opts/fast_render.yaml
```

For better performance, you can use our pre-trained demo models which are trained with all camera views.

```
python run.py --type evaluate --cfg_file configs/exps/im4d/zju/my_313.yaml   --configs configs/components/opts/fast_render.yaml --configs configs/components/opts/render_path/zju_path.yaml exp_name_tag demo
```



</details>

## Training

```
python train_net.py --cfg_file configs/exps/im4d/xx_dataset/xx_scene.yaml
```

Training with multiple GPUs:
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export LOG_LEVEL=WARNING # INFO, DEBUG, WARNING
torchrun --nproc_per_node=$NUM_GPUS train_net.py --cfg_file configs/exps/im4d/xx_dataset/xx_scene.yaml --log_level $LOG_LEVEL distributed True
```


<!-- ## Results -->
<!-- We will release  -->
## Running on the custom dataset

<details> <summary>[TODO] 1. Custom mocap datasets.</summary>
</details>


## Acknowledgements
We would like to acknowledge the following inspring prior work:
- [IBRNet: Learning Multi-View Image-Based Rendering](https://ibrnet.github.io/) (Wang et al.)
- [ENeRF: Efficient Neural Radiance Fields for Interactive Free-viewpoint Video](https://zju3dv.github.io/enerf) (Lin et al.)
- [K-Planes: Explicit Radiance Fields in Space, Time, and Appearance](https://sarafridov.github.io/K-Planes/) (Fridovich-Keil et al.)

Big thanks to [NeRFAcc](https://www.nerfacc.com/) (Li et al.) for their efficient implementation, which has significantly accelerated our rendering.

Recently, in the course of refining our codebase, we have incorporated basic implementations of ENeRF and K-Planes. These additions, although not yet thoroughly tested and aligned with the official codes, could serve as useful resources for further exploration and development.
## Citation

If you find this code useful for your research, please use the following BibTeX entry

```
@inproceedings{lin2023im4d,
  title={High-Fidelity and Real-Time Novel View Synthesis for Dynamic Scenes},
  author={Lin, Haotong and Peng, Sida and Xu, Zhen and Xie, Tao and He, Xingyi and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2023}
}
```
