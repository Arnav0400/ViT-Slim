# One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning

This is an official PyTorch implementation of - One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning.

> [**One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning**](https://arxiv.org/abs//2306.07967)<br>
> [Arnav Chavan](https://sites.google.com/view/arnavchavan/), [Zhuang Liu](https://liuzhuang13.github.io/), [Deepak Gupta](https://dkgupta90.github.io/), [Eric Xing](http://www.cs.cmu.edu/~epxing/), [Zhiqiang Shen](http://zhiqiangshen.com/)<br>MBZUAI, Transmute AI Lab, Meta, CMU

Enhancing Low-Rank Adaptation (LoRA), GLoRA employs a generalized prompt module to optimize pre-trained model weights and adjust intermediate activations, providing more flexibility and capability across diverse tasks and datasets.

<div align=center>
<img width=50% src="method.png"/>
</div>

## Updates
### August '23 : Hugging Face Model

We have open-sourced GLoRA fine-tuned LLaMA-7B on the hugging face hub. Note that GLoRA weights are merged into the base model. 

Huggingface hub link -
[LLaMA-7B-GLoRA-ShareGPT](https://huggingface.co/MBZUAI-LLM/LLaMA-7B-GLoRA-ShareGPT)
### July '23 : Results on LLMs
The table below shows the performance on language tasks with pre-trained **LLaMA-7B** as the backbone.

| Model             | ARC (25-s) | HellaSwag (10-s) | MMLU (5-s) | TruthfulQA (MC) (0-s) | Average |
|-------------------|------------|------------------|------------|-----------------------|---------|
| LLaMA-7B          | 51.0       | 77.8             | 35.7       | 34.3                  | 49.7    |
| Falcon-7B         | 47.9       | 78.1             | 27.8       | 34.3                  | 47.0    |
| Alpaca-LoRA-7B    | 53.5       | 77.3             | 33.8       | 34.8                  | 49.8    |
| Alpaca-GLoRA-7B   | 52.9       | 78.1             | 34.5       | 37.8                  | 50.8    |
| ShareGPT-LoRA-7B  | 51.7       | 77.9             | 36.1       | 39.2                  | 51.2    |
| ShareGPT-GLoRA-7B | 53.2       | 77.4             | 36.2       | 43.9                  | 52.7    |

Additionally, we have implemented GLoRA inside [huggingface/peft](https://github.com/huggingface/peft), which is available in this [fork](https://github.com/Arnav0400/peft). All the experiments in the table above are done using this implementation.

## Getting Started

You will need [Python 3.8](https://www.python.org/downloads) and the packages specified in environment.yml.
We recommend setting up a [conda environment](https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf)
and install the packages there.

## Dataset

Please refer to [NOAH](https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation) to download the dataset. Then move the dataset folders to `data/`.

Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) and place it in the root folder.

## Training
GLoRA follows a two-step process - 1. Supernet Training 2. Evolutionary Search

`LR=1e-4` works very well across datasets, `LORA_RANK` is the rank of LoRA modules across the supernet. In case the `SAVE_DIR` does not exists, a new directory is created. We train supernet for a default of 500 epochs, irrespective of the dataset.

```
python supernet.py
    --dataset DATASET
    --lr LR
    --model 'vit_base_patch16_224_in21k'
    --save_path SAVE_DIR
    --rank LORA_RANK
```

```
python evolution.py
    --dataset DATASET
    --save_path SAVE_DIR
    --load_path supernet training SAVE_DIR
    --param-limits MAX_PARAM(M)
    --rank LORA_RANK
```

## Citation

If you find our project is helpful, please feel free to leave a star and cite our paper:
```BibTeX
@misc{chavan2023oneforall,
      title={One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning}, 
      author={Arnav Chavan and Zhuang Liu and Deepak Gupta and Eric Xing and Zhiqiang Shen},
      year={2023},
      eprint={2306.07967},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
    

## Acknowledgments
Part of the code is borrowed from [FacT](https://github.com/JieShibo/PETL-ViT/tree/main/FacT) and [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer).


<!-- ## License

This project is licensed under the MIT License. -->
