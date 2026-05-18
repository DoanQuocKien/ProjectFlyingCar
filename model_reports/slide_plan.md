## Slide Plan — ProjectFlyingCar (Expanded)

Title: ProjectFlyingCar — Gesture-driven RC car using deep detectors
Subtitle: Real-time single-frame gesture detection (five classes) → HTTP command to ESP32
One-line summary: Trained three detector families on a curated HaGRID subset (five gesture classes). Artifacts: `models/` checkpoints and export folders; runtime pipeline: webcam→detector→decode→map→HTTP to ESP32.

1) Title Slide — quick facts
- Project: ProjectFlyingCar — Gesture-driven RC car using deep detectors.
- Key artifacts: `models/mobilenet_ssd_hagrid_detector.pt` (~3.54M params), `models/resnet18_hagrid_detector.pt` (~11.19M params), and YOLO exports (ONNX / SavedModel / TFLite).
- Short runtime constraints: target latency < 40 ms; reported notebook latencies are MobileNet ≈1.45 ms, ResNet18 ≈3.05 ms, YOLO11n ≈10–14 ms, and YOLO26 ≈12–16 ms depending on runtime and export.
- Visual idea: one hero image of the demo feed or a 2×2 montage of the five gesture classes.

2) Outline with time budget
- Total target talk length 11–12 minutes. Time allocations reflect complexity: Models & math (3.5 min), Training & ablations (2.5 min), Demo (1 min). Each section links to reproducibility material in `models/` and `model_reports/`.

3) Motivation & Problem Statement — measured goals
- Task: map an RGB frame I ∈ R^{H×W×3} to a gesture label y ∈ C={one,peace,three,four,fist} and bounding box B ∈ [0,1]^4. The exact model-specific grid and output-head details are covered in the model slides. End-to-end correctness means the command derived from y and B matches operator intent ≥ 98% on held-out set.
- Operational constraints: single-hand visible, typical indoor lighting; empirical latency target set at T_latency≤40 ms (experimentally we observed MobileNet inference ≈1.45 ms, ResNet18 ≈3.05 ms on our bench). Production target for embedded devices is latency ≤30 ms with quantized models.

4) Contributions & Deliverables — what to expect in the repo
- Checkpoints: full-precision PyTorch `.pt` files for MobileNet and ResNet; YOLO training artifacts under `models/yolo11n/` and `models/yolo26/` including best ONNX and SavedModel exports.
- Repro artifacts: per-epoch CSV training logs, `*_env_info.txt` files listing package versions, and exported confusion matrices and PR/mAP figures in `model_reports/`.
- Visual idea: show a compact artifact map or file-tree screenshot that points to `models/` and `model_reports/`.

5) Dataset — precise composition & preprocessing
- Dataset notation: D = {(I_k,B_k,y_k)}_{k=1}^N with y_k ∈ C and B_k normalized. We used an aggregate of ~8.8k labeled images across train/val/test for detection experiments; specific splits used in each run are recorded in `models/*/*_splits.json`.
- Concrete splits used in classification/detection runs: for ResNet experiments N_train≈6,174; N_val≈1,198; N_test≈1,465. YOLO runs used similar totals (train≈6,185; val≈1,325). Class balance is provided in the per-run `splits` JSON files.
- Preprocessing: images resized to model input size (MobileNet: 320×320, ResNet: 448×448, YOLO: 640×640). Normalization uses backbone-specific mean/std; augmentations included horizontal flip (p=0.5), random crop up to 10% area, brightness/contrast jitter (±15%), and for YOLO mosaic augmentation during training (as configured in the Ultralytics recipe).
- Visual idea: insert one dataset sample montage or label distribution image here so the audience sees the five gestures before the model slides.

6) System Architecture — runtime pipeline with thresholds
- Runtime flow (detailed): capture frame I_t → preprocess (resize + normalize) → detector forward pass D_θ(I_t) → decode box parameters from the model-specific grid (10×10 for MobileNet, 14×14 for ResNet, Ultralytics multi-scale heads for YOLO) → apply score threshold τ_score=0.25 → apply NMS with τ_nms=0.45 → pick top detection by confidence → compute area and map to speed → send HTTP GET to ESP32 endpoint.
- Control parameters used in demos: `TURN_RATIO=0.82`, `LEFT_TRIM=1.0`, `RIGHT_TRIM=0.82`. These were tuned on a small validation bench to compensate for mechanical bias.
- Important implementation detail: mention the notebook decode path explicitly with `decode_batch_predictions`, `decode_single_prediction`, and `decode_and_nms` so the audience sees how raw logits become a command.
- Visual idea: use a pipeline diagram with four boxes: camera, detector, decode/NMS, ESP32 command.

7) Detection math — decode & loss (explicit)
- Prediction vector per cell: $$\\hat y_{i,j}=[o,t_x,t_y,t_w,t_h,z_1,\\dots,z_C]$$ where $o$ is objectness logit, $t_*$ are regression outputs, and $z_c$ are class logits.
- Decoding to absolute coordinates (grid size S and grouping factor G used in our heads): the notebook targets one prediction cell per spatial bin, so the output tensor is $S\\times S\\times(1+4+C)$ and the decoded center coordinates are normalized by the grid size.
  $$c_x=\\frac{j+\\sigma(t_x)}{S},\\quad c_y=\\frac{i+\\sigma(t_y)}{S}$$
  $$w=\\frac{\\sigma(t_w)}{S},\\quad h=\\frac{\\sigma(t_h)}{S}$$
- Probabilities: $p_{obj}=\\sigma(o)$ and class distribution $p_c=\\mathrm{softmax}(z)$. The notebook’s decode helpers then convert a cell index $(i,j)$ into box corners by taking the argmax over objectness, reading the relative offsets at that cell, and rescaling the final box back to the original image size.
- Loss used in training runs (weights recorded in run configs):
  $$\\mathcal{L}=w_{obj}\\,\\mathrm{BCEWithLogits}(o,y_{obj})+w_{box}\\,\\mathrm{SmoothL1}(b,b^*)+w_{cls}\\,\\mathrm{CE}(z,y_{cls}).$$
- For our best runs we set loss weights approximately to $w_{obj}=1.0, w_{box}=5.0, w_{cls}=1.0$; SmoothL1 beta parameter was 1.0. Exact per-run values are in the saved `train_config.yaml` per experiment.

8) Control mapping — from detection to motor command (numbers)
- Compute normalized box area: $area=(w\\cdot h)/S^2$ ∈ [0,1]. We map area→speed nonlinearly:
  $$speed=\\mathrm{round}\\big(s_{min}+area^{0.35}(s_{max}-s_{min})\\big).$$
- Example values used during demos: $s_{min}=20$, $s_{max}=100$ → area=0.36 gives speed≈round(20+0.36^{0.35}·80)≈55.
- Wheel trimming: after computing `active_speed=speed*TURN_RATIO` we applied `LEFT_TRIM` and `RIGHT_TRIM` multiplicative corrections shown above to compensate for hardware variance.

9) Model — MobileNetV3-SSD architecture (compact)
- Backbone: MobileNetV3-Large feature extractor with `mobilenet.features` as the backbone.
- Neck/head: 1×1 adapter from 960→256 channels, Hardswish, adaptive average pooling to 10×10, then an SSD-style detection head with 256→128→10 channels.
- Output tensor: 100 cells total, each cell emits 10 values = 1 objectness logit + 4 box values + 5 class logits.
- Shape story: 320×320 input is compressed to a 10×10 prediction grid, so this slide should emphasize the 32× spatial reduction before the final head.
- Recommended figure: MobileNet confusion matrix or inference visualization from [models/mobilenet_ssd/mobilenet_confusion_matrix.png](../models/mobilenet_ssd/mobilenet_confusion_matrix.png) and [models/mobilenet_ssd/inference_visualization.png](../models/mobilenet_ssd/inference_visualization.png).

10) Model — MobileNetV3-SSD training and results
- Training recipe: optimizer=Adam; phase1 (head only) 6 epochs, lr=1e-2, batch=32; phase2 (full) 35 epochs, lr=5e-4, batch=16; weight decay=1e-4; ReduceLROnPlateau patience=5.
- Notebook behavior: the backbone is frozen first, then thawed for full fine-tuning.
- Best results observed: held-out classification accuracy ≈0.9277, best val ≈0.9291.
- Measured single-frame inference on our bench ≈1.45 ms (GPU), end-to-end (including preprocess+decode) ≈2–3 ms.
- Visual idea: pair the training curve CSV/plot with the confusion matrix so the audience sees convergence and class-level mistakes together.

11) Model — ResNet18-SSD architecture (accuracy-focused)
- Backbone: standard ResNet18 trunk (`children()[:-2]`) with a 1×1 detection head on top.
- Input/output geometry: 448×448 input, 14×14 target grid, 10 channels per cell (1 objectness, 4 box terms, 5 classes).
- Shape story: ResNet18 down-samples by stride-32 overall, so the 448→14 grid is a direct consequence of the backbone resolution reduction.
- Compared to MobileNet, this model trades more parameters for a denser representation and better localization stability.
- Recommended figure: [models/resnet18/resnet18_confusion_matrix.png](../models/resnet18/resnet18_confusion_matrix.png) plus a cropped training-history plot from [models/resnet18/resnet18_training_history.csv](../models/resnet18/resnet18_training_history.csv).

12) Model — ResNet18-SSD training and results
- Training recipe: common run uses Adam lr=1e-4, 20 epochs, batch=12, with ReduceLROnPlateau.
- Best results: test accuracy ≈0.9466, mean IoU ≈0.744.
- Inference time ≈3.05 ms (GPU); recommended for deployment when accuracy is prioritized over throughput.
- This is the slide to show the best class-level balance, including the stronger validation behavior and the more consistent localization metric.
- Visual idea: use both the confusion matrix and one side-by-side prediction example from the notebook visualization cell.

13) YOLO variants — architecture and training pipeline
- Training details: Ultralytics pipeline with imgsz=640, mosaic augment, mixup, multi-scale training.
- Dataset pipeline: the notebooks convert HaGRID annotations to YOLO text format, then split them with the same 0.7/0.15/0.15 logic used by the custom detectors (6,185 train / 1,325 val / 1,327 test).
- Architecture note: the notebooks do not define a custom grid head in Python; they rely on Ultralytics' internal multi-scale detection pyramid, so these should be described as dense anchor-free detectors.
- Training length: yolo11n trained 80 epochs; yolo26 trained 50 epochs.
- Visual idea: use one architecture sketch and one training-result figure only; do not overwhelm the slide with all artifact files.

14) YOLO variants — results and deployment value
- `yolo11n`: mAP@0.5 ≈ 0.9944, mAP@0.5:0.95 ≈ 0.8627, Precision ≈ 0.9917, Recall ≈ 0.9836.
- `yolo26`: mAP@0.5 ≈ 0.9941, mAP@0.5:0.95 ≈ 0.8603, Precision ≈ 0.9941, Recall ≈ 0.9808.
- Exports include ONNX and SavedModel in `models/yolo*/yolo_models/`; these are intended for cross-runtime deployment and quantization experiments.
- This slide is where to emphasize that YOLO is the strongest deployment candidate when export flexibility matters as much as raw metric value.
- Recommended figures: [models/yolo11n/yolo_models/yolo_runs/hagrid_yolo11n/results.png](../models/yolo11n/yolo_models/yolo_runs/hagrid_yolo11n/results.png), [models/yolo11n/yolo_models/yolo_runs/hagrid_yolo11n/confusion_matrix_normalized.png](../models/yolo11n/yolo_models/yolo_runs/hagrid_yolo11n/confusion_matrix_normalized.png), and [models/yolo11n/yolo_models/yolo_runs/hagrid_yolo11n/val_batch0_pred.jpg](../models/yolo11n/yolo_models/yolo_runs/hagrid_yolo11n/val_batch0_pred.jpg) or the analogous yolo26 figures.

15) Quantitative comparison — table & interpretation
- Table columns: model | params | input | latency | primary metric (acc or mAP).
- Key takeaways: MobileNet is best for latency/size (≈1.45 ms, 3.5M params), ResNet18 achieves higher accuracy/IoU (≈3.05 ms, ≈0.9466, 11.2M params), YOLO11n sits around ≈10–14 ms and YOLO26 around ≈12–16 ms depending on export/runtime, while both still deliver very high mAP (≈0.994) and strong deployment flexibility.
- Visual idea: a compact bar chart for latency alongside the table will make this slide much easier to read.

16) Curves & diagnostics — what to inspect
- Include: training/val loss curves, AP per-class, PR curves, and mAP vs epoch. Annotate where learning rate reductions occurred and where overfitting begins (often after epoch ~40 for MobileNet full fine-tune runs under our schedules).
- Visual idea: use one curve panel per model family rather than all plots at once.

17) Qualitative results — examples to present
- Present true positives at high IoU (>0.75), border cases (occlusion / partial finger), and clear failure modes (mis-ordered fingers, extreme tilt). Use side-by-side: image, predicted box & score, ground truth.
- Recommended images: one clean success, one borderline case, and one failure case from the notebook visualizations.

18) Ablation studies — exact experiments and findings
- Suggested ablations we performed: (a) backbone change (MobileNet ↔ ResNet18), (b) grid size / spatial resolution change (320→10×10 vs 448→14×14), (c) augmentations (with/without mosaic), (d) box weight w_box in loss.
- Observations: increasing w_box from 1→5 improved localization (mean IoU +0.03) at slight cost to classification noise; mosaic augmentation increased mAP for YOLO by ~0.01–0.03 depending on class mix.
- Visual idea: one compact table plus one representative ablation figure is enough; keep the slide readable.

19) Runtime considerations & deployment notes
- Exports: ONNX with dynamic axes enabled for batch size 1; TFLite conversion used float16 for better accuracy on-device. Benchmarking strategy: run 1000 inferences, report median latency and 95th percentile jitter.
- Safety: map low-confidence (<0.25) or missing detections to a safe `fist`/`stop` command to avoid unintended motion.
- Visual idea: a small deployment flow or device-stack diagram works better than a dense paragraph here.

20) Demo plan — reproducible commands
```bash
pip install -r requirements.txt  # see requirements.txt or the env info files
python realtime_hand_detector.py --model mobilenet --car-ip http://192.168.137.228 --score-threshold 0.25 --iou-threshold 0.45
```
- During demo show: (1) live feed with overlayed box and speed, (2) terminal HTTP request log, (3) fail-case recovery where connection loss triggers stop.
- Visual idea: a single screenshot of the live overlay is more useful than a long code block on the slide itself.

21) Limitations & failure modes — measured impact
- Lighting/domain gap: performance drops ~3–6% absolute in low-light test subsets. Small, fast gestures near edges produce lower IoU (<0.5) in ~2% of test samples. Latency jitter on CPU-only devices can spike by +20 ms under load.

22) Future work & recommended experiments
- Temporal smoothing: exponential smoothing implemented as $p_t=\\alpha p_{t-1}+(1-\\alpha)p_{frame}$; recommend α≈0.6 for stable responsiveness without oversmoothing.
- Quantization: run Int8 calibration on representative 2k-frame subset; expected size reductions ~4x and inference improvements depending on hardware delegate.

23) Reproducibility — friendly guide for presentations
- What we include: environment snapshots (`*_env_info.txt`), a `requirements.txt` summary, trained checkpoints under `models/`, per-run training logs (`*_training_history.csv`), and the exact training configs used for reported numbers.
- Quick check: run the provided `smoke_test.py` to confirm dependencies and that the chosen model loads correctly (this validates the environment without running full training).
- Want to re-run an experiment? Each model folder (`models/<model>/`) contains a short README and the training config that lists the recipe (optimizer, epochs, batch size, and augmentations). Follow that recipe, or ask and we will provide a single curated command block for the specific model and target device you care about.
- Need help reproducing results end-to-end? We can supply a one-click set of commands (conda + pip + exact run flags) tailored to your machine. Keep this slide high-level during talks and offer the detailed commands in the appendix or a shared Gist.
- Visual idea: a single artifact checklist or QR-style link box is enough.

24) Summary & takeaways — numeric recap
- MobileNet: ~3.54M params, inference ≈1.45 ms, accuracy ≈0.9277 — best latency/size trade.
- ResNet18: ~11.19M params, inference ≈3.05 ms, accuracy ≈0.9466, mean IoU ≈0.744 — best accuracy.
- YOLO11n/26: mAP@0.5 ≈0.994, with notebook latencies around ≈10–14 ms for YOLO11n and ≈12–16 ms for YOLO26 — best for high-precision deployment and cross-runtime exports.
- Visual idea: this slide benefits from a tiny summary chart or icon row, not additional text.

25) Appendix & backup artifacts
- Include links to per-run CSVs, `results.png` plots, `*_env_info.txt`, and the exact training configs in each `models/*/` experiment folder for reproducibility.

Notes: this expanded plan avoids referencing internal draft folders and focuses on reproducible numbers and exact pipeline choices. Replace placeholder values (author, date, ESP32 IP) before presentation.
