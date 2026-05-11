# MobileNet SSD HaGRID Report

## 1. Scope And Evidence

This report is grounded in [main_mobilenet_ssd.ipynb](../main_mobilenet_ssd.ipynb), [models/mobilenet_ssd/mobilenet_training_history.csv](../models/mobilenet_ssd/mobilenet_training_history.csv), [models/mobilenet_ssd/mobilenet_classification_report.json](../models/mobilenet_ssd/mobilenet_classification_report.json), [models/mobilenet_ssd/mobilenet_confusion_matrix.png](../models/mobilenet_ssd/mobilenet_confusion_matrix.png), and [models/mobilenet_ssd/inference_visualization.png](../models/mobilenet_ssd/inference_visualization.png). The saved checkpoint is [models/mobilenet_ssd_hagrid_detector.pt](../models/mobilenet_ssd_hagrid_detector.pt).

The task is 5-class gesture detection on HaGRID: `one`, `peace`, `three`, `four`, `fist`.

The notebook is not a generic classifier. It is a compact detection-style model that combines a pretrained MobileNetV3-Large backbone with a custom grid head, then trains it with a YOLO-like target format on the HaGRID subset.

## 2. Model And Detailed Structure

### 2.1 Backbone And Feature Geometry

The model uses torchvision’s pretrained MobileNetV3-Large as the feature extractor. MobileNetV3-Large is a lightweight convolutional backbone built from a stem convolution followed by inverted residual bottlenecks, depthwise separable convolutions, squeeze-and-excitation blocks, and h-swish activations. The central idea is to reduce compute by separating spatial filtering from channel mixing.

If a standard convolution maps $C_{in}$ channels to $C_{out}$ channels with kernel size $k\times k$, the rough multiply cost per output location is

$$
O(k^2 C_{in} C_{out})
$$

while a depthwise-separable block decomposes this into a depthwise spatial step plus a pointwise channel-mixing step,

$$
O(k^2 C_{in}) + O(C_{in} C_{out})
$$

which is why MobileNetV3 can stay compact enough for real-time gesture detection.

The notebook uses an input resolution of $320 \times 320$ and a supervision grid of $10 \times 10$, so each spatial cell corresponds to roughly a $32 \times 32$ region in the input image.

### 2.2 Detector Head

The backbone features are pooled to a fixed spatial size with adaptive average pooling:

$$
	ext{pool}(F) = \operatorname{AdaptiveAvgPool2d}(F, 10 \times 10)
$$

That pooled map is then processed by a compact SSD-style head:

$$
960 \xrightarrow{1\times1} 256 \xrightarrow{3\times3} 128 \xrightarrow{1\times1} 10
$$

where the final 10 channels are split as

$$
10 = 1\; (p_{obj}) + 4\; (x,y,w,h) + 5\; (class logits)
$$

So each grid cell predicts

$$
\hat{y}_{i,j} = [p_{obj}, t_x, t_y, t_w, t_h, z_1, \dots, z_5].
$$

The design is deliberately small: one extra adapter block sits between the backbone and the detection head, and the head itself is only a couple of convolutions wide. That keeps the model fast while still allowing a detection-specific readout instead of forcing the backbone to behave like a plain classifier.

### 2.3 Output Interpretation

The raw outputs are logits. Objectness and box values are squashed during decoding:

$$
p_{obj} = \sigma(o), \qquad \mathbf{b} = \sigma(\mathbf{t}_{box})
$$

so the predicted box coordinates live in normalized $[0,1]$ space before they are rescaled to pixels.

For a cell $(i,j)$, the decoded center coordinates follow the standard grid-offset form:

$$
c_x = \frac{j + \Delta_x}{G}, \qquad c_y = \frac{i + \Delta_y}{G}
$$

with $G=10$. The width and height are also normalized, so the box can be mapped back to the image by multiplying by the input size $320$.

## 3. Training Pipeline

### 3.1 Data And Target Encoding

The pipeline converts the HaGRID annotations into a dense target tensor of shape $(10,10,6)$ per image. The first channel stores objectness, the next four channels store normalized box coordinates, and the last channel stores the class id.

For an object with normalized center $(c_x,c_y)$, the responsible grid cell is

$$
i = \lfloor c_y G \rfloor, \qquad j = \lfloor c_x G \rfloor
$$

and the relative offset within that cell is

$$
\Delta_x = c_x G - j, \qquad \Delta_y = c_y G - i.
$$

That encoding turns each image into a fixed-size supervision map. Cells with no object remain background cells with zero objectness.

### 3.2 Augmentation And Resolution

Images are resized to $320\times320$. Training data are augmented, while validation and test data stay deterministic so that the reported numbers reflect the actual learned model and not stochastic preprocessing noise.

### 3.3 Two-Phase Freeze-Thaw Optimization

The training schedule is explicitly two phase:

1. Head warm-up: freeze the backbone and train only the adapter plus detection head for 6 epochs at learning rate $10^{-2}$.
2. Full fine-tuning: unfreeze the backbone and train the entire network for 35 epochs at learning rate $5\times10^{-4}$.

The effective training objective is a weighted multi-task loss:

$$
\mathcal{L} = w_{obj}\,\mathcal{L}_{obj} + w_{box}\,\mathcal{L}_{box} + w_{cls}\,\mathcal{L}_{cls}
$$

with $w_{obj}=1$, $w_{box}=5$, and $w_{cls}=1$.

The notebook computes the three parts as:

$$
\mathcal{L}_{obj} = \text{BCEWithLogits}(o, y_{obj})
$$

$$
\mathcal{L}_{box} = \text{SmoothL1}(\mathbf{t}_{box}[y_{obj}=1], \mathbf{b}[y_{obj}=1])
$$

$$
\mathcal{L}_{cls} = \text{CrossEntropy}(z[y_{obj}=1], y_{cls}[y_{obj}=1])
$$

This is the key masking rule: only positive cells contribute to box and class supervision. Background cells still contribute to the objectness term, but they do not force the network to regress boxes where no object exists.

### 3.4 Optimization Behavior

The phase-level optimizer is Adam, and the notebook uses `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)` driven by validation accuracy. That means the learning rate is reduced when validation accuracy stops improving, which is useful in the second phase where the backbone is already close to a usable solution.

The backbone is explicitly kept in evaluation mode during the first phase so batch-norm running statistics do not drift while the feature extractor is frozen. That detail matters because the notebook also records deterministic algorithm warnings for `adaptive_avg_pool2d_backward_cuda`, which means the implementation prioritizes stable reporting over forcing every CUDA path to be bitwise deterministic.

The training history stores one row per epoch with:

$$
	ext{epoch},\;\text{phase},\;\text{train\_loss},\;\text{lr},\;\text{latency\_ms},\;\text{accuracy},\;\text{precision\_macro},\;\text{recall\_macro},\;\text{f1\_macro},\;\text{mean\_iou},\;\text{latency\_mean\_ms}
$$

That makes the report much richer than a single final score, because it exposes both optimization dynamics and runtime behavior.

## 4. Results Report

### 4.1 Overall Performance

The held-out classification report records overall accuracy of $0.9276563677467973$. Macro averages are precision $0.9293921468545344$, recall $0.9268925270160786$, and F1 $0.9278169587905897$.

The model’s best validation accuracy in the CSV is $0.9290566037735849$, reached late in phase 2. That lines up with the final held-out accuracy, so the training curve suggests the model is not badly overfitting at the end.

The full-class performance is strong enough to make this a usable compact detector:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| one | 0.9263565891472868 | 0.9053030303030303 | 0.9157088122605364 | 264 |
| peace | 0.9065040650406504 | 0.9028340080971660 | 0.9046653144016227 | 247 |
| three | 0.8888888888888888 | 0.9523809523809523 | 0.9195402298850575 | 294 |
| four | 0.9607843137254902 | 0.9245283018867925 | 0.9423076923076923 | 265 |
| fist | 0.9644268774703557 | 0.9494163424124513 | 0.9568627450980393 | 257 |

### 4.2 Interpretation Of The Numbers

The best class by F1 is `fist` at $0.9568627450980393$, while `three` has the strongest recall at $0.9523809523809523$. The most visually similar open-hand classes are the ones most likely to be confused, which is exactly what the confusion matrix in [models/mobilenet_ssd/mobilenet_confusion_matrix.png](../models/mobilenet_ssd/mobilenet_confusion_matrix.png) should show.

The history CSV also shows the expected freeze-thaw pattern:

1. Phase 1 climbs quickly from $0.6362$ accuracy to about $0.6657$ accuracy in only 6 epochs because the head is adapting to the dataset while the backbone stays fixed.
2. Phase 2 starts with a jump into the low $0.80$ range after the backbone is unfrozen.
3. Later epochs push validation accuracy into the $0.92$–$0.93$ band, with mean IoU improving from roughly $0.49$ early on to about $0.70$ near the end of training.

The training latency recorded in the CSV is about $216$–$226$ ms per batch in the notebook’s training loop, while the benchmarked mean inference latency is around $1.45$ ms. That split is what you would expect from a compact detector: training is relatively heavier because of backpropagation, but inference is lightweight enough for near-real-time use.

### 4.3 Practical Meaning

This report’s metrics suggest the model is best understood as a compact, deployment-friendly gesture detector rather than a heavyweight accuracy-first model. It is small, uses a minimal detection head, and still reaches strong macro metrics on a five-class hand-gesture task.

## 5. Available Images And Insertable Artifacts

The following ready-made assets are present in the model folder and can be inserted into later reports:

- [models/mobilenet_ssd/mobilenet_confusion_matrix.png](../models/mobilenet_ssd/mobilenet_confusion_matrix.png)
- [models/mobilenet_ssd/inference_visualization.png](../models/mobilenet_ssd/inference_visualization.png)
- [models/mobilenet_ssd/mobilenet_training_history.csv](../models/mobilenet_ssd/mobilenet_training_history.csv)
- [models/mobilenet_ssd/mobilenet_classification_report.json](../models/mobilenet_ssd/mobilenet_classification_report.json)
- [models/mobilenet_ssd/mobilenet_report_summary.json](../models/mobilenet_ssd/mobilenet_report_summary.json)
- [models/mobilenet_ssd/mobilenet_env_info.txt](../models/mobilenet_ssd/mobilenet_env_info.txt)

The saved detector checkpoint is [models/mobilenet_ssd_hagrid_detector.pt](../models/mobilenet_ssd_hagrid_detector.pt).

## 6. Layer / parameter summary (checkpoint-derived)

I loaded `models/mobilenet_ssd_hagrid_detector.pt` and tallied the stored parameter tensors. The checkpoint contains approximately **3,539,898** parameters (total, all tensors).

Top parameter groups (by stored state-dict prefix):

- `backbone`: 2,996,398 params
- `detection_head`: 296,715 params
- `adapt_conv`: 246,785 params

Notes: these counts come from the saved state dictionary; they match the expected breakdown of a MobileNetV3 backbone plus a small adapter and compact detection head. Parameter counts are suitable for comparing model size and for estimating memory/flash requirements for deployment.

## 7. Per-epoch & training summary

- Logged training epochs: **41** (rows present in `models/mobilenet_ssd/mobilenet_training_history.csv`).
- Best validation accuracy found: **0.92906** at epoch **34** (matches the CSV record).
- Recorded training-loop latency (per-batch mean): **209.0 ms**, median **217.1 ms** (logged in the history CSV).

If you want, I can insert the full epoch table (CSV → Markdown) here or embed the training-curve image from the folder.