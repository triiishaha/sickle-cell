# Sickle Cell Classification

A PyTorch pipeline that classifies red blood cell microscopy images as **Negative** or **Positive** for sickle cell disease, comparing three pretrained CNN backbones (ConvNeXt V2, InceptionNeXt, EfficientNetV2) with a full hyperparameter search.

## Overview

Sickle cell disease is diagnosed in part by identifying abnormally shaped ("sickled") red blood cells under a microscope. This project fine-tunes modern pretrained vision backbones on a public blood-cell image dataset to automate that classification, and systematically compares which architecture and training configuration generalizes best.

The notebook covers the full workflow end to end:

1. **Dataset download** — pulls the dataset directly from Kaggle via `kagglehub`.
2. **Data cleaning** — scans every image for corruption and exact duplicates (MD5 hash) and removes them.
3. **Preprocessing** — resizes to 224×224, converts to RGB, normalizes with ImageNet statistics.
4. **Stratified train/test split** — a 70/30 split that preserves the class ratio (the dataset is imbalanced).
5. **Data augmentation** — flips, rotation, random-resized crop, color jitter, and affine shifts applied only to the training set.
6. **Class-imbalance handling** — class-weighted cross-entropy loss and a `WeightedRandomSampler` so minority-class (Negative) samples aren't drowned out.
7. **Model comparison** — three pretrained backbones fine-tuned end to end:
   - ConvNeXt V2 (`convnextv2_tiny`)
   - InceptionNeXt (`inception_next_tiny`)
   - EfficientNetV2 (`efficientnetv2_rw_t`)
8. **Hyperparameter optimization** — an Optuna search narrowed down to 3 hand-picked configs per model (learning rate, batch size, weight decay, dropout, optimizer, scheduler), for 9 total training runs.
9. **Evaluation** — accuracy, F1, Matthews correlation coefficient, balanced accuracy, and AUC-ROC on a held-out test set, with training/validation loss curves for every run.

## Dataset

[Sickle Cell Disease Dataset](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset) (Kaggle, by Florence Tushabe), downloaded automatically at runtime with `kagglehub` — no manual download needed.

| | Raw | After cleanup (4 duplicate pairs removed) |
|---|---|---|
| Negative | 147 | 145 |
| Positive | 844 | 842 |
| **Total** | **991** | **987** |

Stratified 70/30 split: **693 training images** (103 Negative / 590 Positive) and **298 test images** (44 Negative / 254 Positive).

## Results

Each of the three architectures was trained with 3 hyperparameter configurations (9 runs total) and evaluated on the held-out test set:

| Model | Config | Optimizer | Scheduler | Epochs | Test Acc. | F1 | MCC | Balanced Acc. | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| **ConvNeXt V2** | 1 | AdamW | Cosine | 18 | 0.9024 | 0.9397 | 0.7132 | **0.9239** | 0.9682 |
| ConvNeXt V2 | 2 | AdamW | Plateau | 7 | 0.1481 | 0.0000 | 0.0000 | 0.5000 | 0.3494 |
| **ConvNeXt V2** | 3 | SGD | Cosine | 30 | **0.9327** | **0.9605** | **0.7332** | 0.8666 | **0.9686** |
| InceptionNeXt | 1 | AdamW | Cosine | 8 | 0.8822 | 0.9266 | 0.6645 | 0.9027 | 0.9543 |
| InceptionNeXt | 2 | AdamW | Plateau | 8 | 0.8721 | 0.9195 | 0.6557 | 0.9061 | 0.9648 |
| InceptionNeXt | 3 | RMSprop | Cosine | 9 | 0.8653 | 0.9149 | 0.6441 | 0.9022 | 0.9638 |
| EfficientNetV2 | 1 | AdamW | Cosine | 7 | 0.6667 | 0.7614 | 0.3807 | 0.7668 | 0.8096 |
| EfficientNetV2 | 2 | AdamW | Plateau | 8 | 0.5556 | 0.6508 | 0.3171 | 0.7204 | 0.8964 |
| EfficientNetV2 | 3 | SGD | Cosine | 7 | 0.4949 | 0.6231 | 0.0091 | 0.5064 | 0.4799 |

**Best overall: ConvNeXt V2 (Config 3, SGD + cosine annealing)** — 93.27% test accuracy and 0.9686 AUC-ROC. Config 1 (AdamW + cosine) trades a little accuracy for better balanced accuracy (0.9239), making it the better choice if minority-class (Negative) recall matters more than raw accuracy. Some ConvNeXt V2 and EfficientNetV2 runs (Config 2 / Config 3 respectively) collapsed to predicting a single class — evidence of how sensitive this small, imbalanced dataset is to optimizer/scheduler choice.

Full per-run metrics are written to `hyperparameter_comparison_all_models.csv`, and training/validation loss curves for all 9 runs are saved to `loss_curves_all_configs.png` when the notebook is run.

## Repository structure

```
sickle-cell-classification/
├── sickle_cell_classification.ipynb   # full pipeline: data -> training -> evaluation
├── requirements.txt                   # Python dependencies
├── README.md
└── LICENSE
```

## Setup

The notebook was developed for Google Colab (it uses `google.colab.drive` for logging and `google.colab.files` to download the trained checkpoint) but the modeling code is plain PyTorch/timm and runs anywhere with a GPU.

```bash
git clone https://github.com/<your-username>/sickle-cell-classification.git
cd sickle-cell-classification
pip install -r requirements.txt
```

If running outside Colab, remove/replace the `google.colab.drive.mount(...)` and `google.colab.files.download(...)` cells with a local path.

Kaggle dataset download via `kagglehub` requires a Kaggle account; on first run it will prompt for authentication (or read `~/.kaggle/kaggle.json` if present).

## Usage

Open `sickle_cell_classification.ipynb` and run the cells top to bottom:

```bash
jupyter notebook sickle_cell_classification.ipynb
```

1. The dataset downloads automatically.
2. Cleaning, splitting, and augmentation run automatically.
3. All three backbones are instantiated and sanity-checked.
4. The 9 hyperparameter configurations train sequentially (GPU strongly recommended — full run took ~10–20 minutes per config on a Colab GPU).
5. Results, plots, and the trained checkpoint are saved at the end.

## Tech stack

PyTorch · torchvision · timm (pretrained ConvNeXt V2 / InceptionNeXt / EfficientNetV2) · Optuna · scikit-learn · pandas · matplotlib / seaborn · kagglehub

## License

Released under the [MIT License](LICENSE).
