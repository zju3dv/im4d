# Studio Data Preprocess

This file is devoted to guide you for preprocessing the studio data. A typical example is [DNA-Rendering](https://dna-rendering.github.io/) and you can download it from [Download](https://dna-rendering.github.io/inner-download.html). We will take the scene `0023_08` as example.

## File structure

Assume that you have downloaded `0023_08.smc`, `0023_08_annots.smc` and  `0023_08_kinect.smc`, organize them into the following structure.

```
data
└── DNA-rendering
    └── 0023_08
        ├── 0023_08_annots.smc
        ├── 0023_08_kinect.smc
        └── 0023_08.smc
```

## Extract Data

Run the following scripts to extract images, masks and calibration data from smc files.

```bash
export SCENE_PATH=data/DNA-rendering/0023_08
python scripts/preprocess/DNA-rendering/extract_from_smc.py --input ${SCENE_PATH}
```

Then result file structure will be

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


### Mask

If the studio data only have images and calibration data, no masks, then a way to get them is to use [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2). Alternatively, it has been encompassed in [EasyVolCap](https://github.com/zju3dv/EasyVolcap).
