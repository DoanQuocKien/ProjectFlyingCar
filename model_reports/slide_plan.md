## Slide Plan - ProjectFlyingCar CV Presentation

Title: Hand Gesture Detection for Vision-Based Car Control
Subtitle: HaGRID input -> detector output -> decode/NMS -> motion-based speed signal -> detector comparison

This plan keeps the talk focused on Computer Vision. Treat the car as an application context only; avoid hardware planning, ESP32 endpoint details, wheel trim, and long deployment sections.

Recommended length: 12-14 minutes, 16 core slides plus optional appendix.

---

### 1. Problem Definition

Goal: detect one visible hand gesture from an RGB frame.

Model output:

- class label: `one`, `peace`, `three`, `four`, `fist`
- bounding box: normalized hand location
- confidence score

Speaker point: "The CV problem is image-to-detection. The car only consumes the final visual signal."

Visual: one annotated webcam frame with ground-truth box and label.

---

### 2. HaGRID Dataset: Input and Output Form

Dataset form:

`D = {(I_k, B_k, y_k)}`

where:

- `I_k`: RGB image
- `B_k`: hand bounding box, normalized to image size
- `y_k`: gesture class

Filtered class set:

- `one`
- `peace`
- `three`
- `four`
- `fist`

Preprocessing:

- resize image to model input size
- normalize image tensor
- apply augmentation only on training split
- keep validation/test deterministic

Visual: five gesture examples plus one box annotation example.

---

### 3. Target Encoding for Custom Detectors

MobileNet and ResNet use the same YOLOv1-style grid target.

For grid size `S`, each image target is:

`S x S x (1 + 4 + C)`

With `C = 5`:

`S x S x 10`

Each cell predicts:

`[objectness, tx, ty, tw, th, z1, z2, z3, z4, z5]`

Meaning:

- `objectness`: whether this cell owns the hand
- `tx, ty`: center offset inside the responsible grid cell
- `tw, th`: hand box size
- `z1..z5`: class logits

Visual: image -> grid -> responsible cell containing the hand center.

---

### 4. Motion-Based Speed Signal

Include speed calculation, but keep it as temporal CV, not hardware.

Current method:

1. detect hand box in frame `t`
2. compute hand center `c_t`
3. compare with a stable anchor center from the previous accepted movement
4. accumulate small displacement until it is large enough to count
5. compute normalized motion:

`motion = ||c_t - c_(t-1)|| / delta_time`

6. ignore accumulated movement below a deadzone
7. smooth with exponential moving average
8. apply a response curve so slow movement can still change speed
9. map smoothed motion to speed command
10. keep the last stable speed when the hand stops moving, until gesture changes or `stop` is detected

Important contrast:

- old idea: larger box means higher speed
- revised idea: faster hand movement means higher speed

Speaker point: "Moving closer to the camera changes box size, but it should not create speed. Speed comes from temporal hand displacement."

Visual: two consecutive frames with hand centers and an arrow.

---

### 5. Custom Detector Head: YOLOv1-Style Readout

Use this slide before splitting backbones.

Common head idea:

`backbone feature map -> small conv head -> S x S x 10`

For each grid cell:

`p = [o, tx, ty, tw, th, z1, z2, z3, z4, z5]`

Probability terms:

- `p_obj = sigmoid(o)`
- `p_class = softmax(z)`
- score can be treated as objectness combined with class confidence

This is best described as a YOLOv1-style single-stage grid detector, not a full SSD design.

Visual: one feature map cell expanded into the 10-value prediction vector.

---

### 6. MobileNetV3 Flow

Purpose: compact custom detector.

Dimension flow:

`320 x 320 x 3`

`-> MobileNetV3-Large features`

`-> 960 channels`

`-> 1x1 adapter: 960 -> 256`

`-> adaptive pooling: 10 x 10`

`-> conv detection head: 256 -> 128 -> 10`

`-> output: 10 x 10 x 10`

Interpretation:

- 100 prediction cells
- each cell predicts objectness, box, and five gesture logits
- aggressive compression makes it lightweight

Visual: horizontal tensor-shape pipeline.

---

### 7. ResNet18 Flow

Purpose: stronger custom detector with denser grid.

Dimension flow:

`448 x 448 x 3`

`-> ResNet18 convolutional trunk`

`-> 512 x 14 x 14`

`-> 1x1 detection head`

`-> output: 14 x 14 x 10`

Interpretation:

- 196 prediction cells
- same 10-value prediction vector per cell
- denser grid gives more spatial resolution than MobileNet
- more parameters and compute than MobileNet

Visual: horizontal tensor-shape pipeline parallel to the MobileNet slide.

---

### 8. Custom Detector Loss

Use one loss slide for both MobileNet and ResNet.

Training objective:

`L = w_obj * BCEWithLogits(o, y_obj) + w_box * SmoothL1(b, b*) + w_cls * CE(z, y_cls)`

Terms:

- objectness loss: teaches which cell contains the hand
- box loss: teaches localization
- class loss: teaches gesture identity

Project detail:

- box loss is weighted strongly because localization affects detection quality
- class loss is evaluated for the responsible hand cell
- background cells mainly contribute through objectness

Visual: three loss branches merging into one total loss.

---

### 9. Training Setup for MobileNet and ResNet

Keep this slide short. Do not turn it into a model-by-model chapter.

MobileNet training:

- starts from pretrained MobileNetV3 features
- trains head first, then fine-tunes more layers
- smaller input: `320 x 320`

ResNet training:

- starts from pretrained ResNet18 trunk
- trains custom detection head with `448 x 448` input
- larger feature grid: `14 x 14`

Shared points:

- same five HaGRID classes
- same grid-style target encoding
- same loss decomposition
- validation/testing use deterministic preprocessing

Visual: mini table: model, input, grid, training strategy.

---

### 10. Testing and Post-Processing for Custom Models

Raw tensor output is not a detection yet.

Testing pipeline:

1. forward pass gives `S x S x 10`
2. decode each candidate cell into a normalized box
3. apply confidence threshold
4. apply IoU-based NMS
5. keep best box in single-hand mode
6. map class id to gesture label

Decode equations:

`cx = (j + tx) / S`

`cy = (i + ty) / S`

`x1 = cx - w/2`, `y1 = cy - h/2`

`x2 = cx + w/2`, `y2 = cy + h/2`

Implementation references:

- `decode_predictions(...)`
- `_suppress_overlapping_detections_torch(...)`
- `torchvision.ops.nms`

Visual: many candidate boxes -> threshold -> NMS -> one final box.

---

### 11. MobileNet vs ResNet Results

Present as a backbone comparison under the same custom head.

| Model | Params | Input | Grid | Main result |
| --- | ---: | ---: | ---: | --- |
| MobileNetV3 + head | about `3.54M` | `320` | `10 x 10` | accuracy about `0.928` |
| ResNet18 + head | about `11.19M` | `448` | `14 x 14` | accuracy about `0.947`, IoU about `0.744` |

Interpretation:

- MobileNet is the compact fast baseline
- ResNet improves accuracy/localization with higher model cost
- both validate the custom YOLOv1-style head idea

Visual: one compact table plus one prediction sample or confusion matrix.

---

### 12. YOLO11n: Strong Comparative Detector

Position YOLO11n as a stronger off-the-shelf detector family used to compare against the custom MobileNet and ResNet experiments.

Why:

- complete modern one-stage detector
- pretrained backbone and neck
- multi-scale feature maps
- built-in decode and NMS
- strong mAP on the same five-class HaGRID task
- practical export path: ONNX / SavedModel / TFLite

Why the earlier experiments still matter:

- MobileNet and ResNet expose the detection pipeline clearly: target encoding, grid head, loss, decode, and NMS
- their comparison shows how backbone choice changes resolution, parameters, and localization
- YOLO11n then provides a modern reference point for how a mature detector handles the same task

Key results:

- `mAP@0.5` about `0.994`
- `mAP@0.5:0.95` about `0.863`
- precision about `0.992`
- recall about `0.984`

Visual: YOLO11n validation prediction image.

---

### 13. YOLO11n Dimension Flow

Use a high-level shape story. Avoid pretending every internal Ultralytics module is hand-written in the notebook.

Input:

`640 x 640 x 3`

Backbone:

- extracts low-level edges/textures early
- extracts higher-level gesture shape features deeper

Neck:

- fuses multi-scale features so small and large hands can both be detected

Detection scales, conceptually:

- `P3`: about `80 x 80`, fine spatial detail
- `P4`: about `40 x 40`, medium objects
- `P5`: about `20 x 20`, larger/global context

Head output:

- dense box/class predictions from each scale
- decoded into final boxes by the YOLO runtime

Visual: `640 input -> backbone -> neck -> P3/P4/P5 -> detect head`.

---

### 14. YOLO11n Math Behind the Head

YOLO11n is anchor-free in the modern Ultralytics style.

Conceptual prediction at each location:

- class probabilities for the five gestures
- box location relative to the feature-map point
- box edges are modeled more precisely using distribution-style regression

Generic decoding idea:

`box = decode(point, predicted_offsets)`

where a feature-map point predicts distances toward the four box sides:

`l, t, r, b`

Then:

`x1 = px - l`

`y1 = py - t`

`x2 = px + r`

`y2 = py + b`

Compared with the custom YOLOv1-style head:

- custom models predict one box format per grid cell
- YOLO11n predicts across multiple scales
- YOLO11n has stronger built-in assignment, decode, and suppression logic

Visual: feature point with four distances to box edges.

---

### 15. YOLO11n Loss and Testing Pipeline

YOLO11n loss is more advanced than the custom detector loss.

Training loss terms:

- box loss: encourages high-overlap localization
- class loss: predicts the correct gesture class
- DFL loss: Distribution Focal Loss for sharper box-edge localization

Logged terms usually appear as:

`box_loss + cls_loss + dfl_loss`

Testing pipeline:

1. resize/letterbox input
2. forward pass through YOLO11n
3. decode multi-scale predictions
4. threshold by confidence
5. apply NMS
6. return final boxes, scores, and labels

Speaker point: "YOLO11n moves much of the detection engineering from our custom code into a stronger pretrained detector framework."

Visual: YOLO training curves plus one prediction sample.

---

### 16. Final Comparison and Conclusion

Use this as the final slide. Compare all three useful paths.

| Model | Detection style | Input | Output scale | Metric | Runtime feel |
| --- | --- | ---: | --- | --- | --- |
| MobileNetV3 + custom head | YOLOv1-style grid | `320` | `10 x 10` | acc about `0.928` | very light |
| ResNet18 + custom head | YOLOv1-style grid | `448` | `14 x 14` | acc about `0.947`, IoU about `0.744` | light-medium |
| YOLO11n | modern multi-scale YOLO | `640` | `80/40/20` scales | mAP50 about `0.994` | real-time but heavier |

Optional small runtime note for speaker script, not slide text:

- MobileNet is expected to feel fastest because it has the smallest input and compact head.
- ResNet costs more but improves localization.
- YOLO11n costs more per frame but gives the strongest detection result among the compared options.

Final pipeline:

`HaGRID frame -> preprocessing -> detector -> decode/NMS -> gesture box/class -> temporal motion speed`

Final takeaways:

- custom MobileNet/ResNet experiments explain the detection mechanics clearly
- YOLO11n is the strongest comparative detector in this project
- speed calculation belongs in the talk only as a short temporal CV bridge
- avoid hardware details unless the teacher asks during Q&A

Visual: compact end-to-end pipeline plus final comparison table.

---

## Optional Appendix

Use only for Q&A.

### A1. Exact Dataset Splits

- train/validation/test counts
- class distribution
- local dataset paths

### A2. Training Recipes

- optimizer
- epochs
- batch size
- learning-rate schedule
- augmentation details

### A3. Failure Cases

- `peace` vs `three`
- hand near image edge
- motion blur
- low light
- partial occlusion

### A4. Extra Runtime Detail

Only show this if asked:

- old box-area speed mapping
- new motion-center speed mapping
- confidence threshold and NMS threshold
- demo command

---

## What to Remove From the Old Deck

Remove or move to appendix:

- hardware plan
- ESP32 endpoint details
- wheel trim constants
- separate long model-by-model chapters
- repeated latency claims
- file-tree artifact slides
- long deployment/quantization discussion
- too many curves from every training run

Keep:

- dataset shape
- target encoding
- detector head dimensions
- loss
- decode and NMS
- MobileNet flow
- ResNet flow
- training/testing for custom models
- detailed YOLO11n architecture, math, and loss as a comparative modern detector
- final all-model comparison
- one short motion-speed slide
