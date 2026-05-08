# ProjectFlyingCar 🚗🤚

A hand-gesture-controlled RC car powered by real-time deep learning. The system trains a custom object detector on the [HaGRID](https://github.com/hukenovs/hagrid) dataset to recognise five hand gestures, then streams predictions to an ESP32-based car over Wi-Fi.

---

## Overview

| Component | Description |
|-----------|-------------|
| **Training notebooks** | Three Jupyter notebooks that train different detector architectures on HaGRID |
| **Real-time detector** | `realtime_hand_detector.py` — webcam → model inference → optimized car commands |
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

## Quick Start

### 1. Set up Python dependencies

Create or activate your environment, then install the packages used by the runtime and notebooks:

```bash
pip install torch torchvision opencv-python requests keyboard ultralytics
```

If you only plan to use the non-YOLO runtime path, `ultralytics` is optional.

### 2. Confirm the model checkpoints are present

The runtime expects one of these files to exist:

```text
models/resnet18_hagrid_detector.pt
models/mobilenet_ssd_hagrid_detector.pt
models/yolo/yolo_models/yolo_runs/yolo_hagrid_best.pt
```

### 3. Run the real-time detector

Start the webcam pipeline and pick a model when prompted:

```bash
python realtime_hand_detector.py
```

You can also force a model from the command line:

```bash
python realtime_hand_detector.py --model mobilenet --car-ip http://192.168.137.228
python realtime_hand_detector.py --model resnet
python realtime_hand_detector.py --model yolo
```

To choose the YOLO checkpoint variant, use `--yolo-variant` (choices: `yolo26`, `yolo11n`). Example:

```bash
python realtime_hand_detector.py --model yolo --yolo-variant yolo11n
python realtime_hand_detector.py --model yolo --yolo-variant yolo26
```

### 4. Run the keyboard driver

```bash
python drive.py
```

### 5. Run a smoke test

```bash
python smoke_test.py
```

---

## Models

Three detector architectures are supported, each with its own training notebook:

| Model | Notebook | Checkpoint |
|-------|----------|------------|
| **ResNet-18** custom SSD head | `main_resnet18.ipynb` | `models/resnet18_hagrid_detector.pt` |
| **MobileNetV3-Large** transfer SSD | `main_mobilenet_ssd.ipynb` | `models/mobilenet_ssd_hagrid_detector.pt` |
| **YOLOv8** (Ultralytics) | `main_yolo.ipynb` | `models/yolo/yolo_models/yolo_runs/yolo_hagrid_best.pt` |

Note: there are two YOLO variant checkpoints included in `models/`:

- `models/yolo11n/yolo_models/yolo11n_hagrid_best.pt` (alias: `yolo11n`)
- `models/yolo26/yolo_models/yolo_runs/yolo_hagrid_best.pt` (alias: `yolo26`)

### What each model does

- **ResNet-18**: a compact CNN backbone with a custom detection head that predicts objectness, box geometry, and gesture class on a fixed grid.
- **MobileNetV3-Large**: a lighter backbone that trades a small amount of accuracy for better speed on laptops and lower-power GPUs.
- **YOLOv8**: an external detector with the same gesture labels, useful if you want a more standard off-the-shelf detection pipeline.

### Runtime output format

All three models ultimately produce the same gesture labels:

`one`, `peace`, `three`, `four`, `fist`

That means the car-control layer does not change between models. Only the detector changes.

---

## How the Runtime Works

The real-time loop in `realtime_hand_detector.py` follows the same high-level steps regardless of model:

1. Capture a webcam frame.
2. Run the selected detector.
3. Convert the best detection into a gesture label.
4. Map the gesture to a car command.
5. Send the command over HTTP to the ESP32 car.

The command mapping is:

| Gesture | Command |
|---------|---------|
| `one` | `/forward` |
| `peace` | `/right` |
| `three` | `/left` |
| `four` | `/backward` |
| `fist` | `/stop` |

The optimized car sender uses `speedL` and `speedR` instead of a single speed value. That lets the car apply per-wheel trimming and turn reduction before the request is sent.

---

## Math Behind It

### 1. Detection confidence and class choice

For the ResNet and MobileNet paths, the model predicts three things on a grid:

`objectness`, `box`, and `class logits`.

The final gesture is chosen by taking the highest class score at the selected cell. In practice, the code keeps the detection with the best objectness score after NMS.

### 2. Bounding-box area to speed

The runtime turns the detected hand size into speed with a nonlinear mapping:

`speed = round(s_min + area^0.35 * (s_max - s_min))`

where `area` is the normalized box area in `[0, 1]`, and `s_min` / `s_max` are the speed limits.

This means:
- a small hand box gives a lower speed,
- a larger box gives a higher speed,
- the curve is intentionally nonlinear so the speed changes more smoothly near the middle range.

### 3. Wheel-speed optimization for the car

The optimized sender computes wheel speeds like this:

`active_speed = speed * TURN_RATIO` for left/right turns, otherwise `active_speed = speed`.

Then:

`speedL = active_speed * LEFT_TRIM`

`speedR = active_speed * RIGHT_TRIM`

That is why left/right turns are a little slower than straight motion: the turn ratio reduces the active speed before the per-wheel trim is applied.

### 4. Boost behavior in the notebook version

`DoAn.py` includes a distance-based boost rule:

`boost = BOOST_SPEED` if `hand_distance > DISTANCE_THRESHOLD`, otherwise `boost = BASE_SPEED`.

That logic is not currently used by `realtime_hand_detector.py`, which uses the detector box size instead.

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
--model       Model to use: resnet | mobilenet | yolo  (interactive if omitted)
--checkpoint  Path to a custom checkpoint file
--car-ip      ESP32 car IP address
--camera      Webcam index (default: 0)
--score-threshold  Detection confidence threshold (default: 0.30)
--iou-threshold    NMS IoU threshold (default: 0.40)
--mirror      Mirror the webcam preview
```

When the detector is running, the window shows the predicted gesture, command, speed, bounding-box area, and link status.

### Useful run patterns

```bash
# Fastest setup: use the default interactive model selection
python realtime_hand_detector.py

# Force the smaller model for speed
python realtime_hand_detector.py --model mobilenet

# Use the ResNet detector and a different car IP
python realtime_hand_detector.py --model resnet --car-ip http://192.168.1.50

# Flip the webcam preview if your camera feed is mirrored already
python realtime_hand_detector.py --mirror
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

- **Car**: ESP32-based RC car exposing a simple HTTP API (`/forward`, `/backward`, `/left`, `/right`, `/stop`) with wheel-speed query parameters (`speedL`, `speedR`)
- **Host**: Any machine with a webcam and Python 3.11+
- **GPU**: Optional but strongly recommended for smooth real-time inference (~50–100 ms/frame on GPU vs ~1–2 s on CPU)

## Notes on the Codebase

- `realtime_hand_detector.py` is the main runtime entry point.
- `main_resnet18.ipynb`, `main_mobilenet_ssd.ipynb`, and `main_yolo.ipynb` are training notebooks, not runtime scripts.
- `DoAn.py` is a separate gesture-control prototype that uses distance-based boost logic and a direct HTTP sender.
- `smoke_test.py` is useful for checking that the saved checkpoints can still be loaded.

## Troubleshooting

If the detector starts but the car does not move:

1. Check that `--car-ip` points to the ESP32 controller URL.
2. Verify the car firmware still accepts `/forward`, `/backward`, `/left`, `/right`, and `/stop`.
3. Confirm that the selected checkpoint file exists and matches the chosen model.
4. Try `--model mobilenet` first if you want the lowest-latency path.

If you want a deeper explanation of the training notebooks, start with `ENVIRONMENT_SETUP.md` and `CHANGES_SUMMARY.md`.
