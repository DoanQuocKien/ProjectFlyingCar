# ResNet18 HaGRID Report

## 1. Scope And Evidence

This report is grounded in [main_resnet18.ipynb](../main_resnet18.ipynb), [models/resnet18/resnet18_training_history.csv](../models/resnet18/resnet18_training_history.csv), [models/resnet18/resnet18_classification_report.json](../models/resnet18/resnet18_classification_report.json), [models/resnet18/resnet18_confusion_matrix.png](../models/resnet18/resnet18_confusion_matrix.png), [models/resnet18/resnet18_splits.json](../models/resnet18/resnet18_splits.json), [models/resnet18/resnet18_env_info.txt](../models/resnet18/resnet18_env_info.txt), and the checkpoint [models/resnet18/resnet18_hagrid_detector.pt](../models/resnet18/resnet18_hagrid_detector.pt).

The task is the 5-class HaGRID gesture detection problem: `one`, `peace`, `three`, `four`, `fist`.

The notebook records a user-level split of the dataset, with 2,410 train users, 516 validation users, and 517 test users. The resulting record counts are 6,174 training samples, 1,198 validation samples, and 1,465 test samples.

## 2. Model And Detailed Structure

The model reuses a pretrained ResNet18 trunk as the feature extractor and replaces the original classification head with a grid-based detector. The notebook uses $448\times448$ inputs and a $14\times14$ supervision grid, so each image is projected to a dense lattice of 196 candidate cells.

The backbone is formed by taking the pretrained ResNet18 layers up to the final convolutional stage and removing the original global average pool and fully connected classifier. In notebook form, the backbone is built with:

$$
	ext{ResNet18 stem + residual stages} \rightarrow \mathbf{F} \in \mathbb{R}^{B \times 512 \times 14 \times 14} \rightarrow \text{detection head}
$$

The retained backbone output preserves the channel depth of ResNet18's last convolutional block, which is 512 channels. Because the input resolution is $448\times448$, the spatial reduction path yields a $14\times14$ feature map before prediction.

The detection head emits $1+4+C$ channels per cell, so with $C=5$ classes the output per cell is 10 values:

$$
\hat{y}_{i,j} = [p_{obj}, t_x, t_y, t_w, t_h, z_1, z_2, z_3, z_4, z_5]
$$

If the output tensor is shaped $(B, G, G, 10)$ with $G=14$, then the model predicts a dense objectness, box, and class tuple per spatial cell. The notebook then applies sigmoid-style decoding and a confidence filter before NMS.

The residual learning block structure remains the standard ResNet identity mapping:

$$
x_{l+1} = x_l + F(x_l; \theta_l)
$$

This skip-connection formulation stabilizes optimization by keeping gradients able to flow through the identity path. In transfer-learning terms, it lets the pretrained feature extractor contribute low-level and mid-level visual structure while the detection head learns dataset-specific gesture localization.

The notebook also keeps the class vocabulary fixed via:

$$
	ext{TARGET\_CLASSES} = \{\text{one}, \text{peace}, \text{three}, \text{four}, \text{fist}\}
$$

with integer mappings stored in `CLASS_TO_ID` and `ID_TO_CLASS` for training, evaluation, and visualization.

## 3. Training Pipeline

The notebook uses a grid target dataset pipeline similar in spirit to the MobileNet run, but at higher input resolution and with a simpler single-stage training schedule.

$$
i = \lfloor c_y G \rfloor, \qquad j = \lfloor c_x G \rfloor, \qquad G=14
$$

where $(c_x, c_y)$ are the normalized center coordinates of the ground-truth box.

The notebook parses the HaGRID-style annotation structure from the repo-local filtered dataset copy and materializes image records before splitting by user identity. That split strategy is important because it reduces identity leakage across train, validation, and test partitions.

The target cell selection is based on normalized object center coordinates $(c_x,c_y)$:

$$
i = \lfloor c_y G \rfloor, \qquad j = \lfloor c_x G \rfloor, \qquad G=14
$$

The loss decomposition is the same three-part design used by the MobileNet detector, with a stronger box term:

$$
\mathcal{L} = \mathcal{L}_{obj} + 5\,\mathcal{L}_{box} + \mathcal{L}_{cls}
$$

with

$$
\mathcal{L}_{obj} = \text{BCEWithLogits}(o, y_{obj})
$$

$$
\mathcal{L}_{box} = \text{SmoothL1}(\mathbf{t}_{box}[y_{obj}=1], \mathbf{b}[y_{obj}=1])
$$

$$
\mathcal{L}_{cls} = \text{CrossEntropy}(z[y_{obj}=1], y_{cls}[y_{obj}=1])
$$

Only cells with positive objectness supervise the box and class terms, which makes the head behave like a sparse detector rather than a dense classifier.

The training wrapper used in the notebook is `fit_model_with_scheduler(...)` with:

$$
	ext{epochs}=20, \qquad \text{lr}=10^{-4}, \qquad \text{scheduler}=\text{ReduceLROnPlateau}(\text{mode}=\text{max}, \text{factor}=0.5, \text{patience}=3)
$$

The scheduler steps on validation accuracy:

$$
	ext{scheduler.step}(\text{val\_accuracy})
$$

This means the learning rate is reduced only when validation accuracy stops improving, which matches the notebook's focus on classification-style accuracy as the main checkpoint selection criterion.

The inference and evaluation path uses two decoding helpers:

$$
	ext{decode\_batch\_predictions}(\cdot)
$$

for the final report, and

$$
	ext{decode\_and\_nms}(\cdot)
$$

for visualization with non-maximum suppression. The visualization helper `visualize_predictions(...)` rescales normalized coordinates back to pixel space before drawing the predicted boxes.

The training history plot tracks train loss, validation accuracy, and mean IoU over the 20 epochs, so the report can compare convergence speed against localization quality instead of relying on accuracy alone.

## 4. Results Report

The test classification report records accuracy of **0.9466** (94.66%). Macro averages are precision **0.9467**, recall **0.9463**, and F1 **0.9465**.

Per-class metrics:

$$
	ext{one}: P=0.9410, R=0.9679, F1=0.9542
$$

$$
	ext{peace}: P=0.9379, R=0.9412, F1=0.9396
$$

$$
	ext{three}: P=0.9339, R=0.9231, F1=0.9284
$$

$$
	ext{four}: P=0.9585, R=0.9442, F1=0.9513
$$

$$
	ext{fist}: P=0.9624, R=0.9552, F1=0.9588
$$

The strongest class performance is on `fist` with F1 **0.9588**. `one`, `four`, and `fist` all exceed **0.95** F1. `peace` and `three` remain solid at **0.94** and **0.93** F1 respectively, indicating balanced multi-class performance.

The training history shows rapid convergence with the best validation accuracy of **0.9609** at epoch 19. The learning rate schedule stepped down when improvements plateaued, achieving stable high accuracy from epochs 8–20.

The mean IoU metric shows substantial improvement during training, reaching **0.744** by epoch 20 (best at epoch 15: **0.738**). This indicates precise bounding box localization alongside correct class prediction. Inference latency remained stable around **3.05 ms** per frame, confirming the detector's suitability for real-time gesture recognition.

The notebook's export block saved the test report, confusion matrix image, training history CSV, environment snapshot, and split metadata alongside the checkpoint. The report summary confirms all artifact paths and model status.

## 5. Available Images And Insertable Artifacts

The following assets are present and ready for later report insertion:

- [models/resnet18/resnet18_confusion_matrix.png](../models/resnet18/resnet18_confusion_matrix.png)
- [models/resnet18/resnet18_training_history.csv](../models/resnet18/resnet18_training_history.csv)
- [models/resnet18/resnet18_classification_report.json](../models/resnet18/resnet18_classification_report.json)
- [models/resnet18/resnet18_report_summary.json](../models/resnet18/resnet18_report_summary.json)
- [models/resnet18/resnet18_env_info.txt](../models/resnet18/resnet18_env_info.txt)
- [models/resnet18/resnet18_splits.json](../models/resnet18/resnet18_splits.json)
- [models/resnet18_hagrid_detector.pt](../models/resnet18_hagrid_detector.pt)

The most useful figures for a later thesis-style writeup are the confusion matrix, the training history curve plot, and the per-class classification report, because together they show both where the model is accurate and where it still confuses similar gestures.

The saved detector checkpoint is [models/resnet18_hagrid_detector.pt](../models/resnet18_hagrid_detector.pt).

## 6. Layer / parameter summary (checkpoint-derived)

The checkpoint `models/resnet18/resnet18_hagrid_detector.pt` contains approximately **11,191,262** parameters in total.

Top parameter groups (by state-dict prefix):

- `backbone`: 11,186,132 params (pretrained ResNet18 trunk)
- `detection_head`: 5,130 params (custom grid-based detection head)

Note: The vast majority of parameters sit in the ResNet18 backbone (expected for a transfer-learning setup). These counts are useful for deployment size estimates and fair comparisons against the MobileNet and YOLO models in this repository.

## 7. Per-epoch & training summary

- Logged training epochs: **20** (rows present in `models/resnet18/resnet18_training_history.csv`).
- Best validation accuracy found: **0.9609** at epoch **19** (matches the CSV record).
- Best mean IoU achieved: **0.7440** at epoch **20**, prior peak **0.7374** at epoch 16.
- Recorded training-loop latency (per-batch mean): **279.4–284.2 ms**, inference latency per frame: **~3.05 ms**.
- Training loss converged smoothly from **0.660** (epoch 1) to **0.013** (epoch 20).

The model demonstrates strong convergence behavior with validation accuracy plateauing around **0.94+** from epoch 8 onward. The learning rate schedule adjusted at epochs 6, 10–11 when improvements slowed, enabling stable late-stage refinement.