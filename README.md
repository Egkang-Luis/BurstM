# BurstM: Deep Burst Multi-scale SR using Fourier Space with Optical Flow (ECCV 2024)

[EungGu Kang], [Byeonghun Lee], [Sunghoon Im], [Kyong Hwan Jin]

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2304.01194.pdf)

#### News
- **July 01, 2024:** Paper accepted at ECCV 2024 :tada:

<hr />

> *Multiframesuper-resolution(MFSR)achieveshigherperfor- mance than single image super-resolution (SISR), because MFSR lever- ages abundant information from multiple frames. Recent MFSR ap- proaches adapt the deformable convolution network (DCN) to align the frames. However, the existing MFSR suffers from misalignments between the reference and source frames due to the limitations of DCN, such as small receptive fields and the predefined number of kernels. From these problems, existing MFSR approaches struggle to represent high- frequency information. To this end, we propose Deep Burst Multi-scale SR using Fourier Space with Optical Flow (BurstM). The proposed method estimates the optical flow offset for accurate alignment and pre- dicts the continuous Fourier coefficient of each frame for representing high-frequency textures. In addition, we have enhanced the network’s flexibility by supporting various super-resolution (SR) scale factors with the unimodel. We demonstrate that our method has the highest perfor- mance and flexibility than the existing MFSR methods.*
<hr />

## Overall architectures for BurstM
![BurstM_overall_architecture.png](figs/BurstM_overall_architecture.png)


## Comparison with previous models
![BurstM_quantitative_comparison.png](figs/BurstM_quantitative_comparison.png.png)
![BurstM_BurstSR_x4_result.png](figs/BurstM_BurstSR_x4_result.png)
![BurstM_BurstSR_multiscale.png](figs/BurstM_BurstSR_multiscale.png)



## Installation

Our code is based on Ubuntu 22.04, pytorch 2.3.0, CUDA 12.4 (NVIDIA RTX 3090 24GB, sm86) and python 3.10.14.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:
```
conda env create --file environment.yaml
conda activate BurstM
```

## SyntheticBurst
### 1. Download dataset and pre-trained models.

- Download [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
- Download [Pre-trained model for SyntheticBurst dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).

### 2. Download dataset and pre-trained models.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python BurstM_Track_1_train.py --image_dir=<Input DIR>
```

### Evaluation
```
python BurstM_Track_1_evaluation.py
```
