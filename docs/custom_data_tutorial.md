# Tutorial of Custom Data


This tutorial is devoted to help you manage to run Im4D from custom data. Generally there are two kinds of custom data: studio data which has been calibrated like [DNA-Rendering](https://dna-rendering.github.io/), and wild data like [ENeRF-Outdoor dataset](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md) and [Neural 3D](https://github.com/facebookresearch/Neural_3D_Video/releases).

Data needed by im4d are **multi-view synchronized image sequences**, **calibration data** and **a proper bounding box**. Masks are optional.


## Studio Data

> We take [DNA-Rendering](https://dna-rendering.github.io/) as example.

Now we assume that you have already got synchornized image sequences, their corresponding masks and calibration data in easyvolcap format. If not, check this [tutorial](./studio_data_preprocess.md).

1. organize the data in the following file structure. (Suppose using the dataset `DNA-Rendering` and the scene `0023_08` with `60` views and `150` frames.)

    ```
    data
    └── DNA-rendering
        ├── 0023_08
        │   ├── images
        │   │   ├── 00
        │   │   │   ├── 000000.jpg
        │   │   │   ├── ...
        │   │   │   └── 000149.jpg
        │   │   └── 59
        │   │       └── ...
        │   ├── masks
        │   │   ├── 00
        │   │   │   ├── 000000.jpg
        │   │   │   ├── ...
        │   │   │   └── 000149.jpg
        │   │   └── 59
        │   │       └── ...
        │   ├── extri.yml
        │   └── intri.yml
        └── 0013_08
            └── ...
    ```

2. prepare pointclouds into `${SCENE_PATH}/pointcloud/{:05d}.ply` for each frame. For DNA-rendering, there is a script `gen_pcd_from_kinect.py` provided in `dna_rendering_sample_code.zip`.

    ```bash
    export SCENE_PATH=data/DNA-rendering/0023_08
    mkdir -p ${SCENE_PATH}/pointcloud

    # Modify the scripts to generate pointcloud of all frames, not just frame 50
    # We expect that there will be `00001.ply` ~ `00150.ply` in directory `${SCENE_PATH}/pointcloud`
    python gen_pcd_from_kinect.py \
        --smc_rgb_file ${SCENE_PATH}/0023_08.smc \
        --smc_annot_file ${SCENE_PATH}/0023_08_annots.smc \
        --smc_kinect_file ${SCENE_PATH}/0023_08_kinect.smc \
        --out_dir ${SCENE_PATH}/pointcloud
    ```

3. denoise pointcloud. Denoised pointcloud will be stored in `${SCENE_PATH}/pointcloud_denoise`.

    ```bash
    python scripts/preprocess/denoise_pcd.py --input ${SCENE_PATH}/pointcloud
    ```

4. determine scene bounding box. Just pick the maximum and minimum values of the all the denoised pointcloud on xyz coordinates. Alternatively, you can pick by the following script and it will print out the scene bounding box.

    > In DNA-rendering or say renbody data, bounding boxes are determined by the pointcloud of each frame, so no scene bounding box is used actually and this step can be skipped.

    ```bash
    python scripts/preprocess/bbox.py --input ${SCENE_PATH}/pointcloud_denoise
    ```

    - In addition, add some paddings of the bbox if you prefer.


## Wild Data

> We take [ENeRF-Outdoor](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md) as example.

Now we assume you have multi-view synchonrinzed image sequences.

### Calibration

The data are calibrated by [COLMAP](https://colmap.github.io/).

1. set your own environment variable. Choose one frame to calibrate.

    ```bash
    export DATASET_PATH=/path/to/your/dataset/enerf_outdoor
    export SCENE_NAME=actor1
    export FRAME_ID=75
    export WORKSPACE_CALIB=${DATASET_PATH}/${SCENE_NAME}/calibration/${SCENE_NAME}_frame${FRAME_ID}
    ```

2. extract images at the specified frame to `${WORKSPACE_CALIB}/images`.

    ```bash
    python scripts/nerfstudio/easyvv2nerfstudio.py --input ${DATASET_PATH}/${SCENE_NAME} --output ${WORKSPACE_CALIB} --frame_id ${FRAME_ID}
    ```

3. practically, there are several alternative ways to run colmap.

    - By colmap itself.

        1. Run the following scripts.

            > NOTE: we assume all cameras share the same intrinsic pamameters. If you use cameras with various intrinsic parameters, set the option `--ImageReader.single_camera 0`.

            ```bash
            mkdir -p ${WORKSPACE_CALIB}/colmap
            export IMAGE_PATH=${WORKSPACE_CALIB}/images
            export DATABASE_PATH=${WORKSPACE_CALIB}/colmap/database.db
            export CAMERA_MODEL=OPENCV

            # COLMAP
            colmap feature_extractor --image_path $IMAGE_PATH --database_path $DATABASE_PATH --ImageReader.camera_model $CAMERA_MODEL --ImageReader.single_camera 1
            colmap exhaustive_matcher --database_path $DATABASE_PATH
            mkdir -p ${WORKSPACE_CALIB}/colmap/sparse
            colmap mapper --database_path $DATABASE_PATH --image_path $IMAGE_PATH --output_path ${WORKSPACE_CALIB}/colmap/sparse

            # Convert from colmap to easyvv
            python scripts/nerfstudio/colmap2easyvv.py ${WORKSPACE_CALIB}/colmap/sparse/0
            ```

        2. Then the calibration data can be found in `${WORKSPACE_CALIB}/colmap/sparse/0/intri.yml` and `${WORKSPACE_CALIB}/colmap/sparse/0/extri.yml`.

        3. Move them to scene path `${DATASET_PATH}/${SCENE_NAME}`.

            ```bash
            mv ${WORKSPACE_CALIB}/colmap/sparse/0/{intri,extri}.yml ${DATASET_PATH}/${SCENE_NAME}
            ```

    - By [nerfstudio](https://docs.nerf.studio/).
        
        > NOTE: seemingly nerfstudio only supports the case that all cameras share the same intrinsic pamameters. If you use cameras with various intrinsic parameters, use the previous method for calibration.

        1. Install nerfstudio by its manual.
        2. Run the following scripts.

            ```bash
            export IMAGE_PATH=${WORKSPACE_CALIB}/images
            ns-process-data images --skip-image-processing --data ${IMAGE_PATH} --output-dir ${WORKSPACE_CALIB}
            ```

        3. Then colmap related files are in `${WORKSPACE_CALIB}/colmap` and the calibration data can be found in `${WORKSPACE_CALIB}/transforms.json`.

        4. Convert from `transforms.json` to `intri.yml` and `extri.yml`.

            ```bash
            python scripts/nerfstudio/transform2easyvv.py --input ${WORKSPACE_CALIB} --output ${DATASET_PATH}/${SCENE_NAME}
            ```

---

#### Optional | Refine Camera Pose

For some more difficult data, **nerfacto** model in nerfstudio may be of help to refine camera pose.

> If you already have some calibration data in other format, first convert it to `transforms.json` that nerfstudio needs.
>
> - in colmap format
>
> ```bash
> python scripts/nerfstudio/colmap2nerfstudio.py --input ${WORKSPACE_CALIB}/colmap/sparse/0 --output ${WORKSPACE_CALIB}
> ```
>
> - in easyvolcap format
>
> ```bash
> python scripts/nerfstudio/easyvv2nerfstudio.py --input ${DATASET_PATH}/${SCENE_NAME} --output ${WORKSPACE_CALIB} --frame_id ${FRAME_ID} --parse_transform
> ```

1. train nerfacto

    ```bash
    export NS_EXP_NAME=optcam
    export NS_TIME_STAMP=firsttry

    # train on all data
    ns-train nerfacto --output_dir ${WORKSPACE_CALIB}/output --experiment_name ${NS_EXP_NAME} --timestamp ${NS_TIME_STAMP} nerfstudio-data --train-split-fraction 1.0 --data ${WORKSPACE_CALIB}
    ```

2. extract calibration data from the model and convert it to easyvolcap format.

    ```bash
    ns-export cameras --load_config ${WORKSPACE_CALIB}/output/${NS_EXP_NAME}/nerfacto/${NS_TIME_STAMP}/config.yml --output-dir ${WORKSPACE_CALIB}
    python scripts/nerfstudio/refinedtransform2transform.py --input ${WORKSPACE_CALIB} --output ${WORKSPACE_CALIB}
    python scripts/nerfstudio/transform2easyvv.py --input ${WORKSPACE_CALIB} --json_file refined_transforms.json --output ${DATASET_PATH}/${SCENE_NAME}
    ```

3. the calibration data can be found at scene path `${DATASET_PATH}/${SCENE_NAME}`.

---


### Scene Reconstruction

As mentioned in the *Efficient Rendering* part at Section 3.3 in the paper, we can use a similar strategy to improve training process. We utilize the global coarse geometry pretrained by other methods to generate a binary field. Then the binary field serves as a guidance to the sampling of Im4D instead of cascade sampler. In practical, we use [NeRFAcc](https://www.nerfacc.com/) to implement it.

For the scenes that have a clear distinction between foreground and background with only foreground moving, like ENeRF-Outdoor, the general idea is that we first use a static reconstruction method (here we use Instant-NGP) to reconstruct the background, and then use visual hull algorithm to bound a range of foreground. Finally, binary fields can be computed for each frame and NeRFAcc can utilize the binary fields for sampling.


#### Scene Bounding Box

> In background reconstruction and final Im4D, a scene bounding box, or say a global bounding box, is required to bound the scene as a prior. 

If you use colmap or nerfstudio to calibrate, you can extract a pointcloud file to `${WORKSPACE_CALIB}/sparse/0/point_cloud.ply` by the following command. 

```bash
colmap model_converter --input_path ${WORKSPACE_CALIB}/colmap/sparse/0/ --output_path ${WORKSPACE_CALIB}/colmap/sparse/0/point_cloud.ply --output_type PLY
```

- Then just pick the maximum and minimum values of the pointcloud on xyz coordinates. Alternatively, you can pick by the script and it will print out the bbox.

    ```bash
    python scripts/nerfstudio/bbox.py --input ${WORKSPACE_CALIB}/colmap/sparse/0/point_cloud.ply
    ```

- In addition, add some paddings of the bbox if you prefer.

If you refine camera pose by nerfstudio, similarly, you can extract a pointcloud file to `${WORKSPACE_CALIB}/output/exports/pcd/point_cloud.ply`.

```bash
ns-export pointcloud --load-config ${WORKSPACE_CALIB}/output/${NS_EXP_NAME}/nerfacto/${NS_TIME_STAMP}/config.yml --output-dir ${WORKSPACE_CALIB}/output/exports/pcd/ --num-points 100000 --remove-outliers True --normal-method open3d --use-bounding-box False
```

- Since there maybe some outliers, check the pointcloud file with a visualizer and pick a proper bounding box.


#### Background Reconstruction

To train with the volcap dataset, the background images should be organized with the following file structure.

From

```
data
└── enerf_outdoor
    └── actor1
        ├── bkgd
        │   ├── 00.jpg
        │   ├── ...
        │   └── 17.jpg
        └── ...
```

to

```
data
└── enerf_outdoor
    └── actor1
        ├── bkgd
        │   ├── 00
        │   │   └── 00.jpg
        │   ├── ...
        │   └── 17
        │       └── 00.jpg
        └── ...
```

Maybe the following scripts can help.

```shell
#! env bash
for ((i = 0 ; i < 18 ; i++)); do
	if [[ $i -lt 10 ]]; then
		dir="0$i"
	else
		dir=$i
	fi
	mkdir $dir
	mv $dir.jpg $dir/00.jpg
done
```

Here, we use Instant-NGP to reconstruct the background of ENeRF-Outdoor, which has been encompassed in this code.

```bash
# Adjust `bounds`, `intri_file`, `extri_file` and `scene_pcd` corresponding to your own calibration data.
python train_net.py --cfg_file configs/exps/ngp/enerf_outdoor/actor1_background.yaml
```

Export binary fields and bounds and move them to the right place, the directory `${DATASET_PATH}/${SCENE_NAME}/grid/background`.

- Modify `grid_resolution` in `cache_grid.yaml` as you like for the precision. (For background, we use `[512, 512, 512]`)
- Adjust `sigma_thresh` (default: 5.0) in `cache_grid.yaml` to control the threshold of binary fields.
- Adjust `chunk_size_x` and `chunk_size_y` in `cache_grid.yaml` in order to fit in your GPU memory.

```bash
python run.py --type cache_grid --cfg_file configs/exps/ngp/enerf_outdoor/actor1_background.yaml --configs configs/components/opts/cache_grid_background.yaml grid_tag background
mkdir ${DATASET_PATH}/${SCENE_NAME}/grid
mv ${workspace}/result/actor1/ngp/grid/background ${DATASET_PATH}/${SCENE_NAME}/grid
```

#### Foreground Reconstruction

Since we use **visuall hull** algorithm, masks are required, which can be obtained by [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2). Alternatively, it has been encompassed in [EasyVolCap](https://github.com/zju3dv/EasyVolcap).

> For ENeRF-Outdoor dataset, you can find the masks at `bgmtv2`, so you can just rename it to `masks`.

Suppose now you have the masks with the following file structure:

```
data
└── enerf_outdoor
    ├── actor1
    │   ├── calibration
    │   │   └── ...
    │   ├── images
    │   │   ├── 00
    │   │   │   ├── 000000.jpg
    │   │   │   ├── ...
    │   │   │   └── 001199.jpg
    │   │   └── 17
    │   │       └── ...
    │   ├── masks
    │   │   ├── 00
    │   │   │   ├── 000000.jpg
    │   │   │   ├── ...
    │   │   │   └── 001199.jpg
    │   │   └── 17
    │   │       └── ...
    │   ├── extri.yml
    │   └── intri.yml
    └── actor1_4
        └── ...
```

---

##### Optional | Foreground Bounding Box

> For saving memory space, we use binary fields of foreground with the grid resolution `128 * 128 * 128` instead of the same resolution as that of the background (`512 * 512 * 512`). To make it finer, we had better use a smaller bounding box that only bound the foreground instead of the whole scene.

You can obtain foreground bounding box (`<{MIN,MAX}_{X,Y,Z}>` are the global bounding box)

```bash
python scripts/nerfstudio/foreground_bbox.py --input ${DATASET_PATH}/${SCENE_NAME} --frame_sample 0 150 1 --threshold 0.8 --grid_resolution 128 128 128 --min_bound <MIN_X> <MIN_Y> <MIN_Z> --max_bound <MAX_X> <MAX_Y> <MAX_Z> --msk_format {:06d}.jpg
```

- You can find the bbox at `${DATASET_PATH}/${SCENE_NAME}/vhull/{:06d}.npy`.

---

Now we can obtain visual hull by bounding box and the masks.

> Note for the option `--dilation_radius` and `--threshold`. With mask loss supervision metioned below, they are use to control the foreground shape for better foreground depth without jaggies.

- if you want to use foreground bbox

    ```bash
    python scripts/nerfstudio/visual_hull.py --input ${DATASET_PATH}/${SCENE_NAME} --frame_sample 0 150 1 --dilation_radius 3 --threshold 0.8 --grid_resolution 128 128 128 --msk_format {:06d}.jpg
    ```

- if you want to use scene bbox

    ```bash
    python scripts/nerfstudio/visual_hull.py --input ${DATASET_PATH}/${SCENE_NAME} --frame_sample 0 150 1 --dilation_radius 3 --threshold 0.8 --grid_resolution 128 128 128 --use_scene_bbox --min_bound <MIN_X> <MIN_Y> <MIN_Z> --max_bound <MAX_X> <MAX_Y> <MAX_Z> --msk_format {:06d}.jpg
    ```

Then you can find the visual hull files at `${DATASET_PATH}/${SCENE_NAME}/grid/foreground/binarys.npz` and `${DATASET_PATH}/${SCENE_NAME}/grid/foreground/bounds.npz`.

In addition, meshes can be exported at `${DATASET_PATH}/${SCENE_NAME}/grid/foreground/meshes/{:06d}.ply` if you add `--export_mesh` option.

### Run Im4D

Now that you have multi-view synchonrinzed image sequences, their corresponding masks, calibration data and binary fields of `<FRAME_LENGTH>` foregrounds with a binary field of background. Then we can use Im4D that use these binary fields for sampling and we add an additional mask loss to supervise the foreground depth.

Current file structure is shown as below.

```
data
└── enerf_outdoor
    ├── actor1
    │   ├── calibration
    │   │   └── ...
    │   ├── images
    │   │   ├── 00
    │   │   │   ├── 000000.jpg
    │   │   │   ├── ...
    │   │   │   └── 001199.jpg
    │   │   └── 17
    │   │       └── ...
    │   ├── masks
    │   │   ├── 00
    │   │   │   ├── 000000.jpg
    │   │   │   ├── ...
    │   │   │   └── 001199.jpg
    │   │   └── 17
    │   │       └── ...
    │   ├── grid
    │   │   ├── foreground
    │   │   │   ├── binarys.npz
    │   │   │   └── bounds.npz
    │   │   └── background
    │   │       ├── binarys.npz
    │   │       └── bounds.npz
    │   ├── vhull
    │   ├── extri.yml
    │   └── intri.yml
    └── actor1_4
        └── ...
```

Train the network with NeRFAcc.

```bash
# Adjust `bounds`, `intri_file`, `extri_file` and `scene_pcd` corresponding to your own calibration data.
python train_net.py --cfg_file configs/exps/im4d/enerf_outdoor/actor1.yaml
```

Render test view images.

```bash
python run.py --type evaluate --cfg_file configs/exps/im4d/enerf_outdoor/actor1.yaml save_result True
```

Render a video with the selected trajectory.

> You can define your own trajectory path by [nerfstudio](https://docs.nerf.studio/quickstart/viewer_quickstart.html#creating-camera-trajectories).

```bash
python run.py --type evaluate --cfg_file configs/exps/im4d/enerf_outdoor/actor1.yaml --configs configs/components/opts/render_path/enerf_outdoor_nerfstudio_path.yaml
```
