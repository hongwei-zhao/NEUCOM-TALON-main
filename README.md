# TALON

**Decoupled and Distilled: Task-Adaptive LoRA-Teachers with Ensemble Knowledge Transfer for Few-Shot Class-Incremental Learning**

This repository provides the official PyTorch implementation of **TALON**, published in **Neurocomputing**, Volume 708, Article 135200 (2027), available online on September 21, 2026.

**Authors:** Hongwei Zhao, Rui Liu, Yansong Liu, Zhiyuan Zou, Yong Chen

**[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231226025981)** | **[DOI](https://doi.org/10.1016/j.neucom.2026.135200)** | **[arXiv](https://arxiv.org/abs/2609.39390)**

TALON (**T**ask-**A**daptive **LO**RA-Teachers with E**N**semble Knowledge Transfer) is a framework for Few-Shot Class-Incremental Learning (FSCIL). It allocates an independent LoRA-Teacher to each incremental task and consolidates the frozen teachers into a unified LoRA-Student through Ensemble Knowledge Transfer (EKT). Semantic-guided distillation weights teacher contributions using feature-space similarity to balance knowledge retention and adaptation while enabling efficient inference.

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20250904121521.png" style="zoom: 60%;" /></div>

## 📑 Catalogue

- [Dependencies](#-dependencies)
- [Project Structure](#-project-structure)
- [Run experiment](#-run-experiment)
- [Datasets](#-datasets)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## 🌟 Dependencies

1. torch 2.0.1
2. torchvision 0.15.2
3. timm 0.6.12
4. tqdm
5. numpy
6. scipy
7. easydict

The provided configurations use CUDA device `0` by default.

## 📂 Project Structure

```text
NEUCOM-TALON-main/
├── backbone/
│   ├── vit_talon.py          # Vision Transformer with LoRA-Teacher/Student modules
│   └── linears.py            # Classifier layers
├── models/
│   ├── talon.py              # Incremental training and knowledge distillation
│   └── base.py               # Base incremental learner
├── scripts/
│   ├── talon_cifar_B60_Inc5.json
│   ├── talon_cub_B100_Inc10.json
│   ├── talon_inr_B100_Inc10.json
│   └── talon_miniimagenet_B60_Inc5.json
├── utils/
│   ├── data.py               # Dataset paths and preprocessing
│   ├── data_manager.py       # Class ordering and incremental data splits
│   └── inc_net.py            # Backbone and incremental network construction
├── main.py                  # Experiment entry point
├── trainer.py               # Training orchestration and evaluation
├── LICENSE
└── README.md
```

## 🔑 Run experiment

1. Install the dependencies above and prepare the datasets as described in [Datasets](#-datasets).

2. Edit the corresponding configuration in `scripts/` for global settings and hyperparameters. The provided configurations are:

   | Dataset | Configuration | Base classes | Classes per incremental task |
   | --- | --- | --- | --- |
   | CIFAR100 | `scripts/talon_cifar_B60_Inc5.json` | 60 | 5 |
   | CUB200 | `scripts/talon_cub_B100_Inc10.json` | 100 | 10 |
   | ImageNet-R | `scripts/talon_inr_B100_Inc10.json` | 100 | 10 |
   | miniImageNet | `scripts/talon_miniimagenet_B60_Inc5.json` | 60 | 5 |

3. Run from the **repository root**. For example, on CIFAR100:

   ```bash
   python main.py --config=./scripts/talon_cifar_B60_Inc5.json
   ```

   Replace the configuration path with the corresponding file above to run another dataset. Training and evaluation logs are saved under `logs/`.

4. **Hyperparameters**

   You can edit the global parameters and algorithm-specific hyperparameters in the corresponding JSON file.

   These parameters include:

   - **init_cls**: The number of classes in the base task.
   - **increment**: The number of new classes in each subsequent incremental task.
   - **backbone_type**: The backbone network. The provided configurations use `vit_base_patch16_224_talon` (ViT-B/16 pre-trained on ImageNet21K and fine-tuned on ImageNet1K). `vit_base_patch16_224_in21k_talon` uses ImageNet21K pre-training without ImageNet1K fine-tuning.
   - **seed**: The random seeds to run, specified as a list (default: `[1993]`). Class-order shuffling is controlled by **shuffle**, which is `false` in the provided configurations.
   - **kshot**: The number of training samples per new class in incremental tasks (default: `5`).
   - **t_rank** / **s_rank**: The ranks of the LoRA-Teacher and LoRA-Student (default: `8`).
   - **t_lora_positions** / **s_lora_positions**: The LoRA insertion positions, selected from `q`, `k`, `v`, and `mlp` (default: `["mlp"]`).
   - **tuned_epoch** / **inc_tuned_epoch**: The teacher training epochs for the base and incremental tasks.
   - **kd_student_epochs** / **kd_student_lr**: The student distillation epochs and learning rate.
   - **device**: The CUDA device IDs (default: `["0"]`).

## 🔎 Datasets

The repository supports the following four FSCIL benchmarks:

- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**: Google Drive: [link](https://drive.google.com/file/d/1Swpje08SXizLX1QCJzNayIMgvHU1gtVe/view?usp=sharing)
- **miniImageNet**: Google Drive: [link](https://drive.google.com/file/d/1Nq7J-y17cNRDs7bG2h_PTbX1vBYKJ16u/view?usp=sharing)
- **ImageNet-R**: Download from the [official dataset page](https://github.com/hendrycks/imagenet-r) and prepare the train/test split expected by `utils/data.py`.

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

Before training, replace the `[DATA-PATH]` placeholders in the relevant dataset loader in `utils/data.py`. For CIFAR100, this specifies the automatic download directory. For the other datasets, the current loaders expect:

```text
[DATA-PATH]/
├── cub/
│   ├── train/<class_name>/<image files>
│   └── test/<class_name>/<image files>
├── miniimagenet/
│   ├── images/<image files>
│   └── split/
│       ├── train.csv
│       └── test.csv
└── imagenet-r/
    ├── train/<class_name>/<image files>
    └── test/<class_name>/<image files>
```

CUB200 and ImageNet-R are loaded using `torchvision.datasets.ImageFolder`, so images should be organized into class subfolders with matching class names across training and test splits. miniImageNet uses CSV split files with a header followed by `filename,class_id` rows; filenames are relative to `images/`. Its loader assigns labels in order of first appearance, so keep the class order consistent between `train.csv` and `test.csv`. Set the paths to match your local dataset layout.

## 📝 Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{zhao2027talon,
  title   = {Decoupled and distilled: Task-adaptive {LoRA}-teachers with ensemble knowledge transfer for few-shot class-incremental learning},
  author  = {Zhao, Hongwei and Liu, Rui and Liu, Yansong and Zou, Zhiyuan and Chen, Yong},
  journal = {Neurocomputing},
  volume  = {708},
  pages   = {135200},
  year    = {2027},
  doi     = {10.1016/j.neucom.2026.135200},
  url     = {https://www.sciencedirect.com/science/article/pii/S0925231226025981}
}
```

The citation uses the journal issue year (2027); the paper was first available online in September 2026.

## 🤗 Acknowledgments

We thank the following repository for providing helpful components, functions, and datasets for our work.

- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
