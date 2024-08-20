# Robust Domain Adaptive Object Detection with Unified Multi-Granularity Alignment

**[[Project Page]](https://github.com/tiankongzhang/MGA) [[PDF]](https://ieeexplore.ieee.org/abstract/document/10561554)**

This project hosts the code for the implementation of **[Robust Domain Adaptive Object Detection with Unified Multi-Granularity Alignment](https://ieeexplore.ieee.org/abstract/document/10561554)** (PAMI 2024).

## Introduction
Domain adaptive detection aims to improve the generalization of detectors on target domain. To reduce discrepancy in feature distributions between two domains, recent approaches achieve domain adaption through feature alignment in different granularities via adversarial learning. However, they neglect the relationship between multiple granularities and different features in alignment, degrading detection. Addressing this, we introduce a unified multi-granularity alignment (MGA)-based detection framework for domain-invariant feature learning. The key is to encode the dependencies across different granularities including pixel-, instance-, and category-levels simultaneously to align two domains. Specifically, based on pixel-level features, we first develop an omni-scale gated fusion (OSGF) module to aggregate discriminative representations of instances with scale-aware convolutions, leading to robust multi-scale detection. Besides, we introduce multi-granularity discriminators to identify where, either source or target domains, different granularities of samples come from. Note that, MGA not only leverages instance discriminability in different categories but also exploits category consistency between two domains for detection. Furthermore, we present an adaptive exponential moving average (AEMA) strategy that explores model assessments for model update to improve pseudo labels and alleviate local misalignment problem, boosting detection robustness. Extensive experiments on multiple domain adaption scenarios validate the superiority of MGA over other approaches on FCOS and Faster R-CNN detectors.

![](/figs/relations.png)

## Installation 

Check FCOS and Faster-RCNN for installation instructions. 


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
