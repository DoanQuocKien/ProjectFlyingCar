# Quoc Kien — Hand Gesture Detection Pipeline

This folder contains the complete end-to-end hand gesture detection and real-time inference pipeline for the CS231 Project.

---

## Project Overview

The system detects **5 hand gestures** from the [HaGRID](https://github.com/hukenovs/hagrid) dataset using three architectures:

| Gesture | Class ID |
|---------|----------|
| `one`   | 1 |
| `peace` | 2 |
| `three` | 3 |
| `four`  | 4 |
| `fist`  | 5 |

Each model outputs a single best prediction per image:
> **y_i = (o, [x, y, w, h], c)**
> - `o ∈ [0, 1]` — Confidence score
> - `[x, y, w, h]` — Bounding box (normalized center-x, center-y, width, height)
> - `c ∈ {1, …, 5}` — Gesture class label

---

## Folder Structure

```
QuocKien/
├── models/
│   ├── mobilenet_ssd/          # MobileNetV2 weights, plots, reports
│   │   ├── mobilenet_ssd_hagrid_detector.pt
│   │   ├── mobilenet_loss_plot.png
│   │   ├── mobilenet_map50_plot.png
│   │   ├── mobilenet_ssd_combined_plot.png
│   │   ├── mobilenet_classification_report.json
│   │   └── mobilenet_preview.png
│   ├── resnet18/               # ResNet18 weights, plots, reports
│   │   ├── resnet18_hagrid_detector.pt
│   │   ├── resnet18_loss_plot.png
│   │   ├── resnet18_map50_plot.png
│   │   ├── resnet18_combined_plot.png
│   │   ├── resnet18_classification_report.json
│   │   └── resnet_preview.png
│   └── yolo11n/                # YOLOv11n weights and run artifacts
│       └── yolo11n.pt
├── model_reports/              # Markdown performance reports per model
├── data/
│   └── hagrid-sample-30k-384p-5class/   # Local dataset (not tracked by git)
├── mobilenet.ipynb             # MobileNetV2 training notebook (run on Kaggle)
├── resnet.ipynb                # ResNet18 training notebook (run on Kaggle)
├── main_yolo11n.ipynb          # YOLO11n training notebook (run on Kaggle)
├── realtime_hand_detector.py   # Main real-time webcam inference script
├── local_evaluate.py           # Local evaluation: mAP50 + latency from .pt files
├── yolo_eval.py                # YOLO-specific evaluation helper
├── plot_combined.py            # Generates combined training curves from logs
├── plot_mobilenet_log.py       # Generates MobileNet training curves from log
├── plot_resnet_log.py          # Generates ResNet training curves from log
├── mobilenet_log.txt           # Full Kaggle training log for MobileNetV2
├── resnet_log.txt              # Full Kaggle training log for ResNet18
└── utils/                      # Shared utilities
```

---

## Architecture Summary

### MobileNetV2 Detector
- **Backbone**: MobileNetV2 pretrained feature extractor (`features` layers)
- **Pooling**: AdaptiveAvgPool2d (4×4)
- **Head**: `Linear(1280×16, 1024) → ReLU → Dropout(0.2) → Linear(1024, 9)`
- **Output**: 4 box coordinates + 5 class logits

### ResNet18 Detector
- **Backbone**: ResNet18 pretrained feature extractor (all layers except FC)
- **Pooling**: AdaptiveAvgPool2d (4×4)
- **Head**: `Linear(512×16, 1024) → ReLU → Dropout(0.2) → Linear(1024, 9)`
- **Output**: 4 box coordinates + 5 class logits

### Training Strategy (both models)
- **Phase 1** (6 epochs): Freeze backbone, train head only at `lr=1e-3`
- **Phase 2** (35 epochs): Unfreeze full network at `lr=5e-4` with ReduceLROnPlateau

### Loss Function
$$\mathcal{L} = 20 \cdot \mathcal{L}_{SmoothL1}(\hat{b}, b) + 1 \cdot \mathcal{L}_{CE}(\hat{c}, c)$$

---

## Setup

### Requirements
```bash
conda activate science_env
pip install torch torchvision torchmetrics pycocotools matplotlib pillow tqdm
```

### Dataset
Place the HaGRID 5-class dataset at:
```
data/hagrid-sample-30k-384p-5class/
├── ann_train_val/
│   ├── one.json
│   ├── peace.json
│   ├── three.json
│   ├── four.json
│   └── fist.json
└── hagrid_30k/
    ├── train_val_one/
    ├── train_val_peace/
    └── ...
```

---

## Usage

### Real-Time Webcam Inference
```bash
python realtime_hand_detector.py --model mobilenet
python realtime_hand_detector.py --model resnet
python realtime_hand_detector.py --model yolo
```

### Local Evaluation (mAP50 + Latency)
```bash
python local_evaluate.py --model mobilenet
python local_evaluate.py --model resnet
```

### Regenerate Training Plots
```bash
python plot_combined.py        # Combined Loss + mAP50 + IoU for both models
python plot_mobilenet_log.py   # MobileNet separate Loss and mAP50 charts
python plot_resnet_log.py      # ResNet separate Loss and mAP50 charts
```

---

## Results

| Model | Val mAP50 (best) | Classification Accuracy | Parameters |
|-------|-----------------|------------------------|------------|
| MobileNetV2 | 0.530 | 92.8% | ~3.5M |
| ResNet18 | — | ~92%+ | ~11.7M |
| YOLOv11n | — | — | ~2.6M |

> Training logs are in `mobilenet_log.txt` and `resnet_log.txt`.
> Full reports are in `model_reports/`.
