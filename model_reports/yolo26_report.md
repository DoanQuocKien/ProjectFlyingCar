# YOLO26 Folder Report

## 1. Scope And Evidence

This report is grounded in [main_yolo26.ipynb](../main_yolo26.ipynb), [models/yolo26/yolo_models/yolo_runs/yolo_hagrid_metadata.json](../models/yolo26/yolo_models/yolo_runs/yolo_hagrid_metadata.json), [models/yolo26/yolo_models/yolo_runs/hagrid_5class/results.csv](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/results.csv), and the export artifacts under [models/yolo26/yolo_models/yolo_hagrid_best_saved_model/](../models/yolo26/yolo_models/yolo_hagrid_best_saved_model/).

The notebook metadata records the run as `YOLOv11n`, while the notebook itself downloads `yolo26n.pt` during execution. For clarity, this report describes the recorded Ultralytics YOLO pipeline in the `yolo26` folder and preserves the saved metadata values exactly as written.

## 2. Model And Detailed Structure

This run is another Ultralytics one-stage detector for the same 5-class HaGRID subset. Like the YOLO11n run, it predicts dense detection tuples per spatial location:

$$
\hat{y} = [b_x, b_y, b_w, b_h, p_{obj}, p_1, \dots, p_C]
$$

with $C=5$ classes.

The optimized detection objective follows the standard Ultralytics decomposition:

$$
\mathcal{L}_{total} = \lambda_{box}\mathcal{L}_{box} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{dfl}\mathcal{L}_{dfl}
$$

The notebook logs the same 640-pixel input size and the same general augmentation stack as the other YOLO run.

## 3. Training Pipeline

The pipeline mirrors the YOLO11n notebook but with a shorter training schedule.

1. Convert the filtered gesture annotations into YOLO label format.
2. Build train/val/test splits under `yolo_splits`.
3. Train for 50 epochs at $640\times640$ resolution with batch size 16.
4. Validate, benchmark, and export to multiple deployment formats.

The logged trainer settings include:

$$
\text{epochs}=50, \quad \text{batch}=16, \quad \text{imgsz}=640, \quad \text{lr0}=10^{-4}, \quad \text{patience}=15
$$

The notebook reports 6,185 training images and 1,325 validation images during dataset scanning, matching the same filtered HaGRID split structure as the other YOLO notebook.

## 4. Results Report

The saved metadata records the final metrics as:

$$
\text{mAP50}=0.9941394689436667, \qquad \text{mAP50-95}=0.8602907546727807
$$

$$
\text{Precision}=0.9941128068382226, \qquad \text{Recall}=0.9807684400908823
$$

This run is extremely close to the YOLO11n folder’s performance. It is slightly lower on mAP@0.50 and mAP@0.50:0.95, but it preserves excellent precision and recall. The practical difference is small, which suggests the dataset is already close to saturating the smaller YOLO variants.

The exported validation and prediction assets indicate that the model is ready for deployment in ONNX, TensorFlow SavedModel, and TFLite formats. The main useful distinction in this folder is that it gives a second, independently trained Ultralytics checkpoint family to compare against the YOLO11n run.

## 5. Available Images And Insertable Artifacts

The following assets are present in the yolo26 folder and can be inserted into later reports:

- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/results.png](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/results.png)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/confusion_matrix.png](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/confusion_matrix.png)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/confusion_matrix_normalized.png](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/confusion_matrix_normalized.png)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxF1_curve.png](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxF1_curve.png)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxPR_curve.png](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxPR_curve.png)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxP_curve.png](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxP_curve.png)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxR_curve.png](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/BoxR_curve.png)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/labels.jpg](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/labels.jpg)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/train_batch0.jpg](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/train_batch0.jpg)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/train_batch1.jpg](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/train_batch1.jpg)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/train_batch2.jpg](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/train_batch2.jpg)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/val_batch0_labels.jpg](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/val_batch0_labels.jpg)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/val_batch0_pred.jpg](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/val_batch0_pred.jpg)
- [models/yolo26/yolo_models/yolo_runs/hagrid_5class/results.csv](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/results.csv)
- [models/yolo26/yolo_models/yolo_runs/yolo_hagrid_metadata.json](../models/yolo26/yolo_models/yolo_runs/yolo_hagrid_metadata.json)

The export bundle is in [models/yolo26/yolo_models/yolo_hagrid_best_saved_model/](../models/yolo26/yolo_models/yolo_hagrid_best_saved_model/) and includes `saved_model.pb`, `variables/`, `metadata.yaml`, and float16/float32 TFLite exports.

## 6. Concise metrics summary

| Metric | Value |
|---|---:|
| mAP@0.50 | 0.9941394689 |
| mAP@0.50:0.95 | 0.8602907547 |
| Precision | 0.9941128068 |
| Recall | 0.9807684401 |
| Epochs trained | 50 |

These values are taken from `models/yolo26/yolo_models/yolo_runs/yolo_hagrid_metadata.json` and represent the final saved run.

## 7. Embedded figures (recommended for reports)

![Training curves](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/results.png)

![Validation prediction sample](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/val_batch0_pred.jpg)

![Confusion matrix (normalized)](../models/yolo26/yolo_models/yolo_runs/hagrid_5class/confusion_matrix_normalized.png)

## 8. Layer / parameter summary (ONNX-derived)

I parsed the exported ONNX model `models/yolo26/yolo_models/yolo_runs/yolo_hagrid_best.onnx` and computed parameter counts from the initializer tensors (weights). This yields an authoritative parameter total suitable for deployment comparisons.

Top 20 ONNX initializers by parameter count:

- `model.7.conv.weight`: shape=(256, 128, 3, 3), params=294,912
- `model.5.conv.weight`: shape=(128, 128, 3, 3), params=147,456
- `model.20.conv.weight`: shape=(128, 128, 3, 3), params=147,456
- `model.23.cv2.2.0.conv.weight`: shape=(64, 256, 3, 3), params=147,456
- `model.9.cv2.conv.weight`: shape=(256, 512, 1, 1), params=131,072
- `model.8.cv2.conv.weight`: shape=(256, 384, 1, 1), params=98,304
- `model.22.cv1.conv.weight`: shape=(256, 384, 1, 1), params=98,304
- `model.22.cv2.conv.weight`: shape=(256, 384, 1, 1), params=98,304
- `model.23.cv2.1.0.conv.weight`: shape=(64, 128, 3, 3), params=73,728
- `model.8.cv1.conv.weight`: shape=(256, 256, 1, 1), params=65,536
- `model.10.cv1.conv.weight`: shape=(256, 256, 1, 1), params=65,536
- `model.10.cv2.conv.weight`: shape=(256, 256, 1, 1), params=65,536
- `model.13.cv1.conv.weight`: shape=(128, 384, 1, 1), params=49,152
- `model.3.conv.weight`: shape=(64, 64, 3, 3), params=36,864
- `model.8.m.0.m.0.cv1.conv.weight`: shape=(64, 64, 3, 3), params=36,864
- `model.8.m.0.m.0.cv2.conv.weight`: shape=(64, 64, 3, 3), params=36,864
- `model.8.m.0.m.1.cv1.conv.weight`: shape=(64, 64, 3, 3), params=36,864
- `model.8.m.0.m.1.cv2.conv.weight`: shape=(64, 64, 3, 3), params=36,864
- `model.17.conv.weight`: shape=(64, 64, 3, 3), params=36,864
- `model.23.cv2.0.0.conv.weight`: shape=(64, 64, 3, 3), params=36,864

Parameter counts by top-level prefix (ONNX initializers):

- `model`: 2,583,127 params
- `Constant`: 42,027 params
- `Mul`: 7 params
- `onnx::Split`: 5 params

- Total parameters (ONNX initializers): 2,625,166

Notes: The ONNX initializer total (2,625,166) counts stored weight tensors; runtime buffers or serialized non-initializer constants are not included. If you prefer a full layer-by-layer forward-pass table (shapes and params per module), I can attempt a `torchinfo` run on the saved `.pt` checkpoint, but Ultralytics custom modules sometimes require a small wrapper to be `torchinfo`-friendly.

## 9. Per-epoch summary and training timing

- Training epochs: 50 (logged run)
- Total wall-clock training time (logged final timestamp): ~3,587.6 seconds (~1.0 hour)
- Mean epoch duration: ~71.8 seconds

Best-epoch highlights (selected):

- Best mAP@0.50: epoch **48** — mAP@0.50 = **0.99434**
- Best mAP@0.50:0.95: epoch **49** — mAP@0.50:0.95 = **0.85925**
- Final epoch (50) — Precision: **0.98906**, Recall: **0.98341**, mAP@0.50: **0.99427**, mAP@0.50:0.95: **0.85833**

Learning-rate schedule (representative):

- Warmup and peak: lr ≈ 3.7e-4 → peak ≈ **1.07e-3** (around epoch 3)
- Gradual decay to lr ≈ **3.31e-5** by epoch 50

Notes on stability and convergence: the run shows fast warmup (peak by epoch 3), steady improvement in the early epochs, and stable high mAP from ~epoch 30 onward. There is no obvious overfitting signal in validation mAP; precision and recall remain high through the final epochs.

If you want a compact per-epoch table embedded here (for example, epoch / mAP50 / mAP50-95 / precision / recall for all 50 epochs), I can insert it or export it as CSV/Markdown — tell me which format you prefer.

## 10. Deployment & quantization notes

- Input size: 640×640 pixels (static-sized training). Expect the exported models to accept a [1,3,640,640] input tensor or an equivalent dynamic-shape wrapper depending on the export.
- Available exports (use these for downstream performance/size testing):
	- ONNX: `models/yolo26/yolo_models/yolo_runs/yolo_hagrid_best.onnx`
	- TensorFlow SavedModel: `models/yolo26/yolo_models/yolo_hagrid_best_saved_model/`
	- PyTorch checkpoint: `models/yolo26/yolo_models/yolo_hagrid_best.pt`

- Quick quantization suggestions:

	- TFLite post-training quantization (float16): convert the SavedModel to TFLite with `tf.lite.TFLiteConverter` and enable `optimizations = [tf.lite.Optimize.DEFAULT]` and `target_spec.supported_types = [tf.float16]` for a simple size/latency win.

	- ONNX quantization (INT8): use `onnxruntime`'s `quantize_static` with a small calibration dataset (a few hundred samples) to produce an INT8 model for CPUs that benefit from vectorized integer ops.

	- PyTorch static quantization: possible for CPU-only inference but may require module fusion and a prepared calibration loop; for Ultralytics models exported to the `nn.Module` form this can be attempted but will need testing.

- Latency benchmarking tips:

	- Run inference repeatedly (warm-up + measured loop) and report median latency over 100 runs.
	- Test on target hardware (CPU, embedded NPU, mobile) using the appropriate runtime (ONNX Runtime, TFLite Interpreter, or native PyTorch Mobile) to capture realistic numbers.

If you want, I can add a short code snippet for converting the SavedModel to TFLite and running a quick latency check on a sample image.

	### ONNX runtime quick benchmark (measured here)

	I ran a quick ONNX Runtime CPU benchmark on `models/yolo26/yolo_models/yolo_runs/yolo_hagrid_best.onnx` using the included `scripts/onnx_latency_check.py` script.

	- Median latency (100 runs, 10 warmup): **58.472 ms**
	- Mean latency: **90.203 ms**
	- Std: **69.999 ms**
	- Min / Max: **21.339 ms** / **236.890 ms**

	You can reproduce this locally with:

	```bash
	python scripts/onnx_latency_check.py --model models/yolo26/yolo_models/yolo_runs/yolo_hagrid_best.onnx --runs 100 --warmup 10
	```

	Files added to the repository to help with conversion and benchmarking:

	- [scripts/onnx_latency_check.py](scripts/onnx_latency_check.py) — runs ONNX Runtime latency tests.
	- [scripts/convert_savedmodel_to_tflite.py](scripts/convert_savedmodel_to_tflite.py) — helper to convert a SavedModel to TFLite (requires `tensorflow`).