# ProjectFlyingCar 🚗🤚

A hand-gesture-controlled RC car powered by real-time deep learning. The system trains a custom object detector on the [HaGRID](https://github.com/hukenovs/hagrid) dataset to recognise five hand gestures, then streams predictions to an ESP32-based car over Wi-Fi.

---

## Overview

| Component | Description |
|-----------|-------------|
| **Training notebooks** | Three Jupyter notebooks that train different detector architectures on HaGRID |
| **Real-time detector** | `realtime_hand_detector.py` — webcam → model inference → car commands |
| **Keyboard driver** | `drive.py` — manual arrow-key control for the car |

### Gesture → Car command mapping

| Gesture | Command |
|---------|---------|
| ☝️ `one` | Forward |
| ✌️ `peace` | Turn right |
| 🤟 `three` | Turn left |
| 🖖 `four` | Backward |
| ✊ `fist` | Stop |

---

## Models

Three detector architectures are supported, each with its own training notebook:

| Model | Notebook | Checkpoint |
|-------|----------|------------|
| **ResNet-18** custom SSD head | `main_resnet18.ipynb` | `models/resnet18_hagrid_detector.pt` |
| **MobileNetV3-Large** transfer SSD | `main_mobilenet_ssd.ipynb` | `models/mobilenet_ssd_hagrid_detector.pt` |
| **YOLOv8** (Ultralytics) | `main_yolo.ipynb` | `models/yolo/yolo_models/yolo_runs/yolo_hagrid_best.pt` |

---

## Dataset

The notebooks use the **HaGRID Sample 30 k 384p** dataset, filtered to five classes (`one`, `peace`, `three`, `four`, `fist`).

Expected local layout:

```
data/
└── hagrid-sample-30k-384p-5class/
    ├── ann_train_val/
    │   ├── one.json
    │   ├── peace.json
    │   ├── three.json
    │   ├── four.json
    │   └── fist.json
    └── hagrid_30k/
        ├── train_val_one/
        ├── train_val_peace/
        ├── train_val_three/
        ├── train_val_four/
        └── train_val_fist/
```

Download via the Kaggle CLI (see [README_KAGGLEAPI.md](README_KAGGLEAPI.md) for full setup):

```bash
kaggle datasets download -d kinonquc/hagrid-dataset
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install torch torchvision opencv-python requests keyboard
# For YOLO support:
pip install ultralytics
```

### 2. Train a model

Open one of the notebooks in Jupyter (or Colab / Kaggle) and run all cells.  
The notebooks auto-detect the execution environment and set paths accordingly — no manual configuration needed.

```bash
jupyter notebook main_mobilenet_ssd.ipynb
```

### 3. Run real-time hand detection

```bash
python realtime_hand_detector.py --model mobilenet --car-ip http://192.168.137.228
```

Available options:

```
--model       Model to use: resnet18 | mobilenet | yolo  (default: mobilenet)
--checkpoint  Path to a custom checkpoint file
--car-ip      ESP32 car IP address
--no-car      Run detector without sending car commands (demo mode)
--threshold   Detection confidence threshold (default: 0.25)
```

### 4. Manual keyboard control

```bash
python drive.py
```

Use **arrow keys** to drive and **ESC** to quit. Edit `ESP32_IP` in the file if your car's IP differs.

---

## Multi-Environment Support

The training notebooks run unchanged on three platforms:

| Platform | Dataset path | Model output | Workers |
|----------|-------------|--------------|---------|
| **Local** | `./data/hagrid-sample-30k-384p-5class` | `./models` | 0 (Windows) / 2 (Mac/Linux) |
| **Google Colab** | `/content/hagrid_dataset/…` | `/content/models` | 2 |
| **Kaggle** | `/kaggle/input/datasets/…` | `/kaggle/working/models` | 2 |

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed platform instructions and [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) for a full change log.

---

## Project Structure

```
ProjectFlyingCar/
├── main_mobilenet_ssd.ipynb   # MobileNetV3 training notebook
├── main_resnet18.ipynb        # ResNet-18 training notebook
├── main_yolo.ipynb            # YOLOv8 training notebook
├── realtime_hand_detector.py  # Webcam + inference + car control
├── drive.py                   # Keyboard car control
├── smoke_test.py              # CI model instantiation test
├── build_notebook.py          # Notebook build helper
├── data/                      # HaGRID dataset (not tracked by git)
├── models/                    # Saved model checkpoints
├── ENVIRONMENT_SETUP.md       # Platform setup guide
├── CHANGES_SUMMARY.md         # Notebook change log
└── README_KAGGLEAPI.md        # Kaggle CLI reference
```

---

## Hardware

- **Car**: ESP32-based RC car exposing a simple HTTP API (`/forward`, `/backward`, `/left`, `/right`, `/stop?speed=<val>`)
- **Host**: Any machine with a webcam and Python 3.11+
- **GPU**: Optional but strongly recommended for smooth real-time inference (~50–100 ms/frame on GPU vs ~1–2 s on CPU)
