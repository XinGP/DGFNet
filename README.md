# DGFNet

# Multi-Agent Trajectory Prediction with Difficulty-Guided Feature Enhanced Network

## Introduction

The paper will be released before August.
<p align="center">
  <img src="files/DGFNet.png">
</p>

### Argoverse 1(single model)
- **Performance Metrics:**

| Split | brier-minFDE | minFDE | MR | minADE | Param |
|-------|:------------:|:------:|:--:|:------:|:------:|
| Val   | 1.499       | 0.897 | 0.073 | 0.634 | 4.53 |
| Test  | 1.742       | 1.117 | 0.108 | 0.763 | - |

### Argoverse 1(ensemble model)
- **Performance Metrics:**

| Split | brier-minFDE | minFDE | MR | minADE |
|-------|:------------:|:------:|:--:|:------:|
| Test  | 1.693       | 1.110 | 0.107 | 0.752 | 

## Qualitative Results

* On Argoverse 1 motion forecasting dataset
<p align="center">
  <img src="files/AV1-1.png" width = "250"/>
  <img src="files/AV1-2.png" width = "250"/>
  <img src="files/AV1-3.png" width = "250"/>
  <img src="files/AV1-4.png" width = "250"/>
</p>

* On Argoverse 2 motion forecasting dataset
<p align="center">
  <img src="files/AV2-1.png" width = "250"/>
  <img src="files/AV2-2.png" width = "250"/>
  <img src="files/AV2-3.png" width = "250"/>
  <img src="files/AV2-4.png" width = "250"/>
</p>

----


## Gettting Started

### Install dependencies
- Create a new conda virtual env
```
conda create --name DGFNet python=3.8
conda activate DGFNet
```

- Install PyTorch according to your CUDA version. We recommend CUDA >= 11.1, PyTorch >= 1.8.0.
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

- Install Argoverse 1 APIs, please follow [argoverse-api](https://github.com/argoai/argoverse-api).

- Install other dependencies
```
pip install scikit-image IPython tqdm ipdb tensorboard
```

### Train from scratch

- Preprocess full Argoverse 1 motion forecasting dataset using the script:
```
sh scripts/argo_preproc_all.sh
```

- Launch training using the script:
```
sh scripts/DGFNet_av1_train.sh
```

- For model evaluation, please refer to the following scripts:
```
sh scripts/DGFNet_av1_eval.sh
```


## Acknowledgment
We would like to express sincere thanks to the authors of the following packages and tools:
- [Simpl](https://github.com/HKUST-Aerial-Robotics/SIMPL)
- [ADAPT](https://https://github.com/gorkaydemir/ADAPT)

## License
This repository is licensed under [MIT license](https://github.com/XinGP/DGFNet/blob/main/LICENSE).
