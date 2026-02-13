# GolfDB: A Video Database for Golf Swing Sequencing

The code in this repository is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). 

## Introduction
GolfDB is a high-quality video dataset created for general recognition applications 
in the sport of golf, and specifically for the task of golf swing sequencing. 

This repo contains a simple PyTorch implemention of the SwingNet baseline model presented in the 
[paper](https://arxiv.org/abs/1903.06528).
The model was trained on split 1 **without any data augmentation** and achieved an average PCE of 71.5% (PCE
of 76.1% reported in the paper is credited to data augmentation including horizontal flipping and affine 
transformations). 

If you use this repo please cite the GolfDB paper:
```
@InProceedings{McNally_2019_CVPR_Workshops,
author = {McNally, William and Vats, Kanav and Pinto, Tyler and Dulhanty, Chris and McPhee, John and Wong, Alexander},
title = {GolfDB: A Video Database for Golf Swing Sequencing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```

## Dependencies
* [PyTorch](https://pytorch.org/)

## Getting Started
Run [generate_splits.py](./data/generate_splits.py) to convert the .mat dataset file to a dataframe and 
generate the 4 splits.

### Train
* I have provided the preprocessed video clips for a frame size of 160x160 (download 
[here](https://drive.google.com/file/d/1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ/view?usp=sharing)). 
Place 'videos_160' in the [data](./data/) directory. 
If you wish to use a different input configuration you must download the YouTube videos (URLs provided in 
dataset) and preprocess the videos yourself. I have provided [preprocess_videos.py](./data/preprocess_videos.py) to
help with that.

* Download the MobileNetV2 pretrained weights from this [repository](https://github.com/tonylins/pytorch-mobilenet-v2) 
and place 'mobilenet_v2.pth.tar' in the root directory. 

* Run [train.py](train.py)

#### Optimized training (speed + accuracy)
`train.py` now includes:
* Mixed precision training (AMP) and gradient scaling
* cuDNN benchmark + high matmul precision on CUDA
* DataLoader optimizations (`pin_memory`, `persistent_workers`, `prefetch_factor`)
* OneCycleLR scheduler + gradient clipping
* Periodic validation and automatic best-checkpoint saving (`models/swingnet_best.pth.tar`)

Environment knobs (all optional):
* `GOLFDB_NUM_WORKERS` (default: `6`)
* `GOLFDB_USE_AMP` (default: `1`)
* `GOLFDB_USE_COMPILE` (default: `0`)
* `GOLFDB_PIN_MEMORY` (default: `1`)
* `GOLFDB_PERSISTENT_WORKERS` (default: `1`)
* `GOLFDB_PREFETCH_FACTOR` (default: `4`)
* `GOLFDB_FREEZE_LAYERS` (default: `0`)
* `GOLFDB_MAX_GRAD_NORM` (default: `1.0`)
* `GOLFDB_LOG_EVERY` (default: `10`)
* `GOLFDB_EVAL_INTERVAL` (default: `100`)
* `GOLFDB_EVAL_NUM_WORKERS` (default: `min(max(num_workers, 1), 4)`)
* `GOLFDB_EVAL_DISP` (default: `0`)
* `GOLFDB_DATALOADER_TIMEOUT_S` (default: `60`, forced to `0` when `GOLFDB_NUM_WORKERS=0`)

Sample commands:

High-performance training on GPU:
```bash
GOLFDB_NUM_WORKERS=8 \
GOLFDB_USE_AMP=1 \
GOLFDB_USE_COMPILE=1 \
GOLFDB_PIN_MEMORY=1 \
GOLFDB_PERSISTENT_WORKERS=1 \
GOLFDB_PREFETCH_FACTOR=4 \
GOLFDB_FREEZE_LAYERS=0 \
GOLFDB_EVAL_INTERVAL=100 \
GOLFDB_EVAL_NUM_WORKERS=4 \
GOLFDB_LOG_EVERY=20 \
python train.py
```

Debug data loading / path issues:
```bash
GOLFDB_DEBUG_TRAIN=1 \
GOLFDB_DEBUG_DATALOADER=1 \
GOLFDB_NUM_WORKERS=0 \
python train.py
```

### Evaluate
* Train your own model by following the steps above or download the pre-trained weights 
[here](https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view?usp=sharing). Create a 'models' directory
if not already created and place 'swingnet_1800.pth.tar' in this directory.

* Run [eval.py](eval.py). If using the pre-trained weights provided, the PCE should be 0.715.  

Evaluate the best checkpoint from training:
```bash
python eval.py
```
`eval.py` uses `models/swingnet_best.pth.tar` by default when present. Override with:
```bash
GOLFDB_EVAL_CKPT=models/swingnet_1800.pth.tar python eval.py
```

### Test your own video
* Follow steps above to download pre-trained weights. Then in the terminal: `python3 test_video.py -p test_video.mp4`

* **Note:** This code requires the sample video to be cropped and cut to bound a single golf swing. 
I used online video [cropping](https://ezgif.com/crop-video) and [cutting](https://online-video-cutter.com/) 
tools for my golf swing video. See test_video.mp4 for reference.

Good luck!
