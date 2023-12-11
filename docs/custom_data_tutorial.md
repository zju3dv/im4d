# Tutorial of Custom Data


This tutorial is devoted to help you manage to run im4D from custom data. Generally there are two kinds of custom data: studio data which has been calibrated like [DNA-Rendering](https://dna-rendering.github.io/), and wild data like [ENeRF-Outdoor dataset](https://github.com/zju3dv/ENeRF/blob/master/docs/enerf_outdoor.md) and [Neural 3D](https://github.com/facebookresearch/Neural_3D_Video/releases).

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

[TODO] tutorial on wild data.
