# Robust Domain Adaptive Object Detection with Unified Multi-Granularity Alignment ---FCOS


## Installation 

Check [INSTALL.md] for installation instructions. 

The implementation of our anchor-free detector is heavily based on FCOS ([\#f0a9731](https://github.com/tianzhi0549/FCOS/tree/f0a9731dac1346788cc30d5751177f2695caaa1f)).


## Dataset

refer to  EveryPixelMatters(https://github.com/chengchunhsu/EveryPixelMatters) for all details of dataset construction.


## Training

To reproduce our experimental result, we recommend training the model by following steps.

Let's take Cityscapes -> Foggy Cityscapes as an example.

export PYTHONPATH=$PWD:$PYTHONPATH

- Using VGG-16 as backbone with 2 GPUs

export CUDA_VISIBLE_DEVICES=0,1

[first stage]

 ```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2434 tools/train_net_da_s0.py --config-file ./configs/RMGA/city/VGG16/S0/da_ga_cityscapes_VGG_16_FPN_4x-s0.yaml OUTPUT_DIR /your_path/ MODEL.ISSAMPLE True
```

[second stage]

 ```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2434 tools/train_net_da_s1.py --config-file ./configs/RMGA/city/VGG16/S1/da_ga_cityscapes_VGG_16_FPN_4x-s1.yaml OUTPUT_DIR /your_path/ MODEL.ISSAMPLE True
```


## Evaluation

The trained model can be evaluated by the following command.

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port=2300 tools/test_net_db.py --config-file ./configs/RMGA/city/VGG16/S1/da_ga_cityscapes_VGG_16_FPN_4x-s1.yaml MODEL.WEIGHT ./model_rs.pth
```

**Environments**

- Hardware
  - 2 NVIDIA 3090 GPUs

- Software
  - PyTorch 1.3.1
  - Torchvision 0.4.2
  - CUDA 11.4



## Citations

Please consider citing our paper in your publications if the project helps your research.

```
@ARTICLE{10561554,
  author={Zhang, Libo and Zhou, Wenzhang and Fan, Heng and Luo, Tiejian and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Robust Domain Adaptive Object Detection With Unified Multi-Granularity Alignment}, 
  year={2024},
  volume={},
  number={},
  pages={1-18}
}
```
