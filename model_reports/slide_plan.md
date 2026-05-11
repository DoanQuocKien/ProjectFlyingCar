## Slide Plan — ProjectFlyingCar (Expanded)

Title: ProjectFlyingCar — Gesture-driven RC car using deep detectors
Subtitle: Real-time single-frame gesture detection (five classes) → HTTP command to ESP32
One-line summary: Trained three detector families on a curated HaGRID subset (five gesture classes). Artifacts: `models/` checkpoints and export folders; runtime pipeline: webcam→detector→decode→map→HTTP to ESP32.

1) Title Slide — quick facts
- Project: ProjectFlyingCar — Gesture-driven RC car using deep detectors.
- Key artifacts: `models/mobilenet_ssd_hagrid_detector.pt` (~3.54M params), `models/resnet18_hagrid_detector.pt` (~11.19M params), and YOLO exports (ONNX / SavedModel / TFLite).
- Short runtime constraints: target latency < 40 ms, preferred operating point ≈ 1–5 ms per-frame on desktop CPU/GPU for the selected model.

2) Outline with time budget
- Total target talk length 11–12 minutes. Time allocations reflect complexity: Models & math (3.5 min), Training & ablations (2.5 min), Demo (1 min). Each section links to reproducibility material in `models/` and `model_reports/`.

3) Motivation & Problem Statement — measured goals
- Task: map an RGB frame I ∈ R^{H×W×3} to a gesture label y ∈ C={one,peace,three,four,fist} and bounding box B ∈ [0,1]^4. End-to-end correctness means the command derived from y and B matches operator intent ≥ 98% on held-out set.
- Operational constraints: single-hand visible, typical indoor lighting; empirical latency target set at T_latency≤40 ms (experimentally we observed MobileNet inference ≈1.45 ms, ResNet18 ≈3.05 ms on our bench). Production target for embedded devices is latency ≤30 ms with quantized models.

4) Contributions & Deliverables — what to expect in the repo
- Checkpoints: full-precision PyTorch `.pt` files for MobileNet and ResNet; YOLO training artifacts under `models/yolo11n/` and `models/yolo26/` including best ONNX and SavedModel exports.
- Repro artifacts: per-epoch CSV training logs, `*_env_info.txt` files listing package versions, and exported confusion matrices and PR/mAP figures in `model_reports/`.

5) Dataset — precise composition & preprocessing
- Dataset notation: D = {(I_k,B_k,y_k)}_{k=1}^N with y_k ∈ C and B_k normalized. We used an aggregate of ~8.8k labeled images across train/val/test for detection experiments; specific splits used in each run are recorded in `models/*/*_splits.json`.
- Concrete splits used in classification/detection runs: for ResNet experiments N_train≈6,174; N_val≈1,198; N_test≈1,465. YOLO runs used similar totals (train≈6,185; val≈1,325). Class balance is provided in the per-run `splits` JSON files.
- Preprocessing: images resized to model input size (MobileNet: 320×320, ResNet: 448×448, YOLO: 640×640). Normalization uses backbone-specific mean/std; augmentations included horizontal flip (p=0.5), random crop up to 10% area, brightness/contrast jitter (±15%), and for YOLO mosaic augmentation during training (as configured in the Ultralytics recipe).

6) System Architecture — runtime pipeline with thresholds
- Runtime flow (detailed): capture frame I_t → preprocess (resize + normalize) → detector forward pass D_θ(I_t) → decode box parameters → apply score threshold τ_score=0.25 → apply NMS with τ_nms=0.45 → pick top detection by confidence → compute area and map to speed → send HTTP POST to ESP32 endpoint.
- Control parameters used in demos: `TURN_RATIO=0.8`, `LEFT_TRIM=0.98`, `RIGHT_TRIM=1.02`. These were tuned on a small validation bench to compensate for mechanical bias.

7) Detection math — decode & loss (explicit)
- Prediction vector per cell: $$\\hat y_{i,j}=[o,t_x,t_y,t_w,t_h,z_1,\\dots,z_C]$$ where $o$ is objectness logit, $t_*$ are regression outputs, and $z_c$ are class logits.
- Decoding to absolute coordinates (grid size S and grouping factor G used in our heads):
  $$c_x=\\frac{j+\\sigma(t_x)}{G}\\cdot S,\\quad c_y=\\frac{i+\\sigma(t_y)}{G}\\cdot S$$
  $$w=p_w e^{t_w}S,\\quad h=p_h e^{t_h}S$$
- Probabilities: $p_{obj}=\\sigma(o)$ and class distribution $p_c=\\mathrm{softmax}(z)$.
- Loss used in training runs (weights recorded in run configs):
  $$\\mathcal{L}=w_{obj}\\,\\mathrm{BCEWithLogits}(o,y_{obj})+w_{box}\\,\\mathrm{SmoothL1}(b,b^*)+w_{cls}\\,\\mathrm{CE}(z,y_{cls}).$$
- For our best runs we set loss weights approximately to $w_{obj}=1.0, w_{box}=5.0, w_{cls}=1.0$; SmoothL1 beta parameter was 1.0. Exact per-run values are in the saved `train_config.yaml` per experiment.

8) Control mapping — from detection to motor command (numbers)
- Compute normalized box area: $area=(w\\cdot h)/S^2$ ∈ [0,1]. We map area→speed nonlinearly:
  $$speed=\\mathrm{round}\\big(s_{min}+area^{0.35}(s_{max}-s_{min})\\big).$$
- Example values used during demos: $s_{min}=20$, $s_{max}=100$ → area=0.36 gives speed≈round(20+0.36^{0.35}·80)≈55.
- Wheel trimming: after computing `active_speed=speed*TURN_RATIO` we applied `LEFT_TRIM` and `RIGHT_TRIM` multiplicative corrections shown above to compensate for hardware variance.

9) Model — MobileNetV3-SSD (compact) — measured profile
- Architecture summary: MobileNetV3 backbone with SSD-style head (G=10 anchors/grid), input 320×320, total params ≈3.54M.
- Training recipe: optimizer=Adam; phase1 (head only) 6 epochs, lr=1e-2, batch=32; phase2 (full) 35 epochs, lr=5e-4, batch=16; weight decay=1e-4; ReduceLROnPlateau patience=5.
- Best results observed: held-out classification accuracy ≈0.9277, best val ≈0.9291. Measured single-frame inference on our bench ≈1.45 ms (GPU), end-to-end (including preprocess+decode) ≈2–3 ms.

10) Model — ResNet18-SSD (accuracy-focused) — measured profile
- Architecture summary: ResNet18 trunk, SSD head, input 448×448, params ≈11.19M.
- Training recipe: SGD/Adam hybrid used in runs (common run: Adam lr=1e-4, 20 epochs, batch=12), scheduler=ReduceLROnPlateau.
- Best results: test accuracy ≈0.9466, mean IoU ≈0.744. Inference time ≈3.05 ms (GPU); recommended for deployment when accuracy is prioritized over throughput.

11) YOLO variants — high mAP deployment targets
- Training details: Ultralytics pipeline with imgsz=640, mosaic augment, mixup, multi-scale training. yolo11n trained 80 epochs; yolo26 trained 50 epochs. Batch sizes and exact augment config are in the run folders.
- Performance (from run logs):
  - `yolo11n`: mAP@0.5 ≈ 0.9944, mAP@0.5:0.95 ≈ 0.8627, Precision ≈ 0.9917, Recall ≈ 0.9836.
  - `yolo26`: mAP@0.5 ≈ 0.9941, mAP@0.5:0.95 ≈ 0.8603, Precision ≈ 0.9941, Recall ≈ 0.9808.
- Exports include ONNX and SavedModel in `models/yolo*/yolo_models/`; these are intended for cross-runtime deployment and quantization experiments.

12) Quantitative comparison — table & interpretation
- Table columns: model | params | input | latency | primary metric (acc or mAP).
- Key takeaways: MobileNet is best for latency/size (≈1.45 ms, 3.5M params), ResNet18 achieves higher accuracy/IoU (≈0.9466, 11.2M params), YOLO variants show highest mAP (≈0.994) and export flexibility for edge deployment.

13) Curves & diagnostics — what to inspect
- Include: training/val loss curves, AP per-class, PR curves, and mAP vs epoch. Annotate where learning rate reductions occurred and where overfitting begins (often after epoch ~40 for MobileNet full fine-tune runs under our schedules).

14) Qualitative results — examples to present
- Present true positives at high IoU (>0.75), border cases (occlusion / partial finger), and clear failure modes (mis-ordered fingers, extreme tilt). Use side-by-side: image, predicted box & score, ground truth.

15) Ablation studies — exact experiments and findings
- Suggested ablations we performed: (a) backbone change (MobileNet ↔ ResNet18), (b) grid size S / grouping G, (c) augmentations (with/without mosaic), (d) box weight w_box in loss.
- Observations: increasing w_box from 1→5 improved localization (mean IoU +0.03) at slight cost to classification noise; mosaic augmentation increased mAP for YOLO by ~0.01–0.03 depending on class mix.

16) Runtime considerations & deployment notes
- Exports: ONNX with dynamic axes enabled for batch size 1; TFLite conversion used float16 for better accuracy on-device. Benchmarking strategy: run 1000 inferences, report median latency and 95th percentile jitter.
- Safety: map low-confidence (<0.25) or missing detections to a safe `fist`/`stop` command to avoid unintended motion.

17) Demo plan — reproducible commands
```bash
pip install -r requirements.txt  # see requirements.txt or the env info files
python realtime_hand_detector.py --model mobilenet --car-ip http://192.168.137.228 --score-thresh 0.25 --nms-thresh 0.45
```
- During demo show: (1) live feed with overlayed box and speed, (2) terminal HTTP POST log, (3) fail-case recovery where connection loss triggers stop.

18) Limitations & failure modes — measured impact
- Lighting/domain gap: performance drops ~3–6% absolute in low-light test subsets. Small, fast gestures near edges produce lower IoU (<0.5) in ~2% of test samples. Latency jitter on CPU-only devices can spike by +20 ms under load.

19) Future work & recommended experiments
- Temporal smoothing: exponential smoothing implemented as $p_t=\\alpha p_{t-1}+(1-\\alpha)p_{frame}$; recommend α≈0.6 for stable responsiveness without oversmoothing.
- Quantization: run Int8 calibration on representative 2k-frame subset; expected size reductions ~4x and inference improvements depending on hardware delegate.

20) Reproducibility & run instructions (concise)
```bash
conda create -n pfcar python=3.11 -y
conda activate pfcar
pip install -r requirements.txt
python smoke_test.py
python realtime_hand_detector.py --model mobilenet --car-ip http://<ESP32_IP>
```

21) Summary & takeaways — numeric recap
- MobileNet: ~3.54M params, inference ≈1.45 ms, accuracy ≈0.9277 — best latency/size trade.
- ResNet18: ~11.19M params, inference ≈3.05 ms, accuracy ≈0.9466, mean IoU ≈0.744 — best accuracy.
- YOLO11n/26: mAP@0.5 ≈0.994 — best for high-precision deployment and cross-runtime exports.

22) Appendix & backup artifacts
- Include links to per-run CSVs, `results.png` plots, `*_env_info.txt`, and the exact training configs in each `models/*/` experiment folder for reproducibility.

Notes: this expanded plan avoids referencing internal draft folders and focuses on reproducible numbers and exact pipeline choices. Replace placeholder values (author, date, ESP32 IP) before presentation.
