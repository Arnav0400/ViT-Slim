# GLoRA

<div align=center>
<img width=80% src="method.png"/>
</div>

## Getting Started

You will need [Python 3.8](https://www.python.org/downloads) and the packages specified in environment.yml.
We recommend setting up a [conda environment](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf)
and installing the packages there.

## Dataset

Please refer to [NOAH](https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation) to download the dataset. Then move the dataset folders to `data/`.

Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) and place it in the root folder.

## Training
GLoRA follows a two step process - 1. Supernet Training 2. Evolutionary Search

```
python supernet.py --dataset DATASET
python evolution.py --dataset DATASET
```

<!-- ## Citation
Please cite our paper in your publications if it helps your research.

    @inproceedings{chavan2022vision,
      title={Vision Transformer Slimming: Multi-Dimension Searching in Continuous Optimization Space},
      author={Chavan, Arnav and Shen, Zhiqiang and Liu, Zhuang and Liu, Zechun and Cheng, Kwang-Ting and Xing, Eric},
      journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    } -->
    

## Acknowledgments
Part of the code is borrowed from [FacT](https://github.com/JieShibo/PETL-ViT/tree/main/FacT) and [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer).


<!-- ## License

This project is licensed under the MIT License. -->
