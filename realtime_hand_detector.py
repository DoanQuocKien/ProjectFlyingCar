from __future__ import annotations

import argparse
import os
import requests
import time
import threading
from pathlib import Path
from queue import Empty, Queue

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from torchvision import models
    from torchvision.ops import nms
except Exception as exc:
    raise RuntimeError(
        "This script requires torchvision with NMS support installed."
    ) from exc


DEFAULT_CLASSES = ["one", "peace", "three", "four", "fist"]
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "resnet18_hagrid_detector.pt"
DEFAULT_MOBILENET_CHECKPOINT = Path(__file__).resolve().parent / "models" / "mobilenet_ssd_hagrid_detector.pt"
DEFAULT_YOLO_CHECKPOINT = Path(__file__).resolve().parent / "models" / "yolo" / "yolo_models" / "yolo_runs" / "yolo_hagrid_best.pt"
# Added explicit YOLO variant checkpoints
DEFAULT_YOLO11N_CHECKPOINT = Path(__file__).resolve().parent / "models" / "yolo11n" / "yolo_models" / "yolo11n_hagrid_best.pt"
DEFAULT_YOLO26_CHECKPOINT = Path(__file__).resolve().parent / "models" / "yolo26" / "yolo_models" / "yolo_runs" / "yolo_hagrid_best.pt"
DEFAULT_CAR_IP = "http://192.168.137.93"
BASE_SPEED = 150
BOOST_SPEED = 250
TURN_RATIO = 0.82
LEFT_TRIM = 1.0
RIGHT_TRIM = 0.82
MIN_CAR_SPEED = 127
MAX_CAR_SPEED = 255
DEFAULT_MOTION_LOW = 0.04
DEFAULT_MOTION_HIGH = 0.55
DEFAULT_MOTION_DEADZONE = 0.06
DEFAULT_MOTION_SMOOTHING = 0.50
DEFAULT_MOTION_CURVE = 0.65
COMMAND_MAP = {
    "one": "forward",
    "peace": "right",
    "three": "left",
    "four": "backward",
    "fist": "stop",
}

# Per-class size bias (optional tuning)
CLASS_SIZE_WEIGHTS = {
    "one": 1.05,
    "peace": 1.03,
    "three": 1.015,
    "four": 1.0,
    "fist": 1.0,
}


class ResNetLocalization(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 4 + num_classes)
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        out = self.head(feat)
        box_preds = torch.sigmoid(out[:, :4])
        cls_logits = out[:, 4:]
        return box_preds, cls_logits


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")

    class_to_id = checkpoint.get("class_to_id") or {name: idx for idx, name in enumerate(DEFAULT_CLASSES)}
    id_to_class = {idx: name for name, idx in class_to_id.items()}
    num_classes = len(class_to_id)
    img_size = int(checkpoint.get("img_size", 448))
    grid_size = int(checkpoint.get("grid_size", 14))

    model = ResNetLocalization(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, id_to_class, img_size, grid_size


# --- MobileNet SSD transfer model (from notebook) ---
class MobileNetLocalization(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Linear(1280 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 4 + num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        out = self.head(feat)
        box_preds = torch.sigmoid(out[:, :4])
        cls_logits = out[:, 4:]
        return box_preds, cls_logits


def load_mobilenet_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")

    class_to_id = checkpoint.get("class_to_id") or {name: idx for idx, name in enumerate(DEFAULT_CLASSES)}
    id_to_class = {idx: name for name, idx in class_to_id.items()}
    num_classes = len(class_to_id)
    img_size = int(checkpoint.get("img_size", 320))
    grid_size = int(checkpoint.get("grid_size", 10))

    model = MobileNetLocalization(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, id_to_class, img_size, grid_size


# --- YOLO wrapper (ultralytics) ---
class YOLOWrapper:
    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        img_size: int = 416,
        iou_threshold: float = 0.45,
        max_det: int = 1,
    ):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics YOLO is required for --model=yolo") from exc
        self.model = YOLO(str(checkpoint_path))
        self.device = device
        self.img_size = img_size
        self.iou_threshold = iou_threshold
        self.max_det = max(1, max_det)
        self.device_arg = 0 if device.type == "cuda" else "cpu"
        self.use_half = device.type == "cuda"
        try:
            self.model.to(self.device_arg)
        except Exception:
            pass

    @staticmethod
    def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
        area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def _suppress_overlapping_detections(self, xyxy: np.ndarray, confs: np.ndarray, cls: np.ndarray):
        if len(xyxy) <= 1:
            return xyxy, confs, cls

        order = np.argsort(-confs)
        keep: list[int] = []

        for idx in order:
            candidate_box = xyxy[idx]
            if all(self._box_iou(candidate_box, xyxy[kept_idx]) <= self.iou_threshold for kept_idx in keep):
                keep.append(int(idx))

        keep_indices = np.asarray(keep, dtype=int)
        return xyxy[keep_indices], confs[keep_indices], cls[keep_indices]

    def predict_frame(self, frame_bgr: np.ndarray, score_threshold: float = 0.25):
        results = self.model(
            frame_bgr,
            imgsz=self.img_size,
            conf=score_threshold,
            iou=self.iou_threshold,
            agnostic_nms=True,
            max_det=self.max_det,
            device=self.device_arg,
            half=self.use_half,
            verbose=False,
        )
        if len(results) == 0:
            return None
        res = results[0]
        if not hasattr(res, 'boxes') or res.boxes is None or len(res.boxes) == 0:
            return None
        xyxy = res.boxes.xyxy.cpu().numpy()  # pixels
        confs = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        # Apply additional score filtering as backup
        keep_mask = confs >= score_threshold
        if not keep_mask.any():
            return None
        xyxy = xyxy[keep_mask]
        confs = confs[keep_mask]
        cls = cls[keep_mask]
        if len(xyxy) > 1:
            xyxy, confs, cls = self._suppress_overlapping_detections(xyxy, confs, cls)
        h, w = frame_bgr.shape[:2]
        boxes_norm = torch.from_numpy(xyxy / np.array([w, h, w, h], dtype=float)).float()
        scores = torch.from_numpy(confs).float()
        labels = torch.from_numpy(cls).long()
        return {"boxes": boxes_norm, "scores": scores, "labels": labels}


def load_yolo_model(checkpoint_path: Path, device: torch.device, img_size: int = 416, max_det: int = 1):
    # Return a wrapper and dummy id_to_class mapping
    wrapper = YOLOWrapper(checkpoint_path, device, img_size=img_size, max_det=max_det)
    id_to_class = {i: name for i, name in enumerate(DEFAULT_CLASSES)}
    return wrapper, id_to_class, wrapper.img_size, None


def _box_iou_torch(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two boxes (normalized coords 0-1)."""
    x1 = torch.max(box_a[0], box_b[0])
    y1 = torch.max(box_a[1], box_b[1])
    x2 = torch.min(box_a[2], box_b[2])
    y2 = torch.min(box_a[3], box_b[3])

    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    inter_area = inter_w * inter_h

    area_a = torch.clamp(box_a[2] - box_a[0], min=0.0) * torch.clamp(box_a[3] - box_a[1], min=0.0)
    area_b = torch.clamp(box_b[2] - box_b[0], min=0.0) * torch.clamp(box_b[3] - box_b[1], min=0.0)
    union = area_a + area_b - inter_area
    union = torch.clamp(union, min=1e-6)
    return inter_area / union


def _suppress_overlapping_detections_torch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply class-agnostic suppression to overlapping detections."""
    if len(boxes) <= 1:
        return boxes, scores, labels

    order = torch.argsort(scores, descending=True)
    keep: list[int] = []

    for idx in order:
        candidate_box = boxes[idx]
        if all(
            _box_iou_torch(candidate_box, boxes[kept_idx]).item() <= iou_threshold
            for kept_idx in keep
        ):
            keep.append(int(idx.item()))

    keep_indices = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]


def decode_predictions(
    obj_logits: torch.Tensor,
    box_preds: torch.Tensor,
    cls_logits: torch.Tensor,
    grid_size: int | None,
    score_threshold: float,
    iou_threshold: float,
    detection_mode: str = "single",
    max_det: int = 1,
):
    if grid_size is None:
        grid_size = int(obj_logits.shape[-1])

    obj_probs = torch.sigmoid(obj_logits)
    batch_predictions = []

    for batch_index in range(obj_probs.size(0)):
        probs = obj_probs[batch_index].reshape(-1)
        keep_mask = probs > score_threshold

        if not keep_mask.any():
            batch_predictions.append(None)
            continue

        filtered_probs = probs[keep_mask]
        flat_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        cell_y = flat_indices // grid_size
        cell_x = flat_indices % grid_size

        rel_boxes = box_preds[batch_index, cell_y, cell_x]
        cx = (cell_x.float() + rel_boxes[:, 0]) / grid_size
        cy = (cell_y.float() + rel_boxes[:, 1]) / grid_size
        half_w = rel_boxes[:, 2] / 2.0
        half_h = rel_boxes[:, 3] / 2.0

        boxes = torch.stack(
            [cx - half_w, cy - half_h, cx + half_w, cy + half_h],
            dim=1,
        ).clamp(0.0, 1.0)

        kept_indices = nms(boxes, filtered_probs, iou_threshold)
        class_ids = torch.argmax(cls_logits[batch_index, cell_y, cell_x], dim=1)

        # Apply class-agnostic overlap suppression
        boxes_nms = boxes[kept_indices]
        scores_nms = filtered_probs[kept_indices]
        labels_nms = class_ids[kept_indices]
        boxes_final, scores_final, labels_final = _suppress_overlapping_detections_torch(
            boxes_nms, scores_nms, labels_nms, iou_threshold
        )

        if len(scores_final) > 0:
            order = torch.argsort(scores_final, descending=True)
            boxes_final = boxes_final[order]
            scores_final = scores_final[order]
            labels_final = labels_final[order]

        if detection_mode == "single" and len(scores_final) > 1:
            boxes_final = boxes_final[:1]
            scores_final = scores_final[:1]
            labels_final = labels_final[:1]
        elif detection_mode == "multi" and max_det > 0 and len(scores_final) > max_det:
            boxes_final = boxes_final[:max_det]
            scores_final = scores_final[:max_det]
            labels_final = labels_final[:max_det]

        batch_predictions.append(
            {
                "boxes": boxes_final,
                "scores": scores_final,
                "labels": labels_final,
            }
        )

    return batch_predictions


def preprocess_frame(frame_bgr: np.ndarray, img_size: int, device: torch.device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def draw_predictions(
    frame_bgr: np.ndarray,
    predictions,
    id_to_class: dict[int, str],
    frame_w: int,
    frame_h: int,
):
    if predictions is None:
        cv2.putText(
            frame_bgr,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    if len(predictions["boxes"]) == 0:
        cv2.putText(
            frame_bgr,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
        x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
        x1 = int(x1 * frame_w)
        y1 = int(y1 * frame_h)
        x2 = int(x2 * frame_w)
        y2 = int(y2 * frame_h)

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        class_name = id_to_class.get(int(label), str(int(label)))
        label_text = f"{class_name} {float(score):.2f}"

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = max(25, y1 - 10)
        cv2.putText(
            frame_bgr,
            label_text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame_bgr


def compute_speed_from_box_size(
    box,
    min_speed: int = MIN_CAR_SPEED,
    max_speed: int = MAX_CAR_SPEED,
) -> int:
    x1, y1, x2, y2 = np.asarray(box.detach().cpu().numpy(), dtype=np.float32)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    box_area = max(0.0, min(1.0, width * height))
    # Simple direct mapping: map box-area curve into [min_speed, max_speed].
    # This avoids a two-stage remap that could compress the observed range.
    curve = box_area ** 0.35
    mapped = min_speed + curve * (max_speed - min_speed)
    speed = int(round(mapped))
    return max(min_speed, min(max_speed, speed))


def get_box_center(box) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box.detach().cpu().numpy(), dtype=np.float32)
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def get_box_center_from_array(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


class HandMotionSpeedTracker:
    """Convert hand center motion into a stable car speed command."""

    def __init__(
        self,
        min_speed: int = MIN_CAR_SPEED,
        max_speed: int = MAX_CAR_SPEED,
        low_velocity: float = DEFAULT_MOTION_LOW,
        high_velocity: float = DEFAULT_MOTION_HIGH,
        deadzone: float = DEFAULT_MOTION_DEADZONE,
        smoothing: float = DEFAULT_MOTION_SMOOTHING,
        curve: float = DEFAULT_MOTION_CURVE,
    ):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.low_velocity = max(0.0, low_velocity)
        self.high_velocity = max(self.low_velocity + 1e-6, high_velocity)
        self.deadzone = max(0.0, deadzone)
        self.smoothing = max(0.0, min(1.0, smoothing))
        self.curve = max(0.2, min(2.0, curve))
        self.prev_center: np.ndarray | None = None
        self.prev_time: float | None = None
        self.anchor_center: np.ndarray | None = None
        self.anchor_time: float | None = None
        self.smoothed_velocity = 0.0
        self.current_speed = min_speed

    def reset(self):
        self.prev_center = None
        self.prev_time = None
        self.anchor_center = None
        self.anchor_time = None
        self.smoothed_velocity = 0.0
        self.current_speed = self.min_speed

    def update_center(self, center: np.ndarray, timestamp: float) -> tuple[int, float]:
        center = np.asarray(center, dtype=np.float32)
        if self.prev_center is None or self.prev_time is None:
            self.prev_center = center
            self.prev_time = timestamp
            self.anchor_center = center
            self.anchor_time = timestamp
            self.current_speed = self.min_speed
            return self.current_speed, self.smoothed_velocity

        if self.anchor_center is None or self.anchor_time is None:
            self.anchor_center = self.prev_center
            self.anchor_time = self.prev_time

        dt = max(1e-3, timestamp - self.anchor_time)
        distance = float(np.linalg.norm(center - self.anchor_center) / np.sqrt(2.0))
        raw_velocity = distance / dt
        if distance < self.deadzone:
            self.prev_center = center
            self.prev_time = timestamp
            return self.current_speed, self.smoothed_velocity

        alpha = self.smoothing
        self.smoothed_velocity = alpha * raw_velocity + (1.0 - alpha) * self.smoothed_velocity
        ratio = (self.smoothed_velocity - self.low_velocity) / (self.high_velocity - self.low_velocity)
        ratio = max(0.0, min(1.0, ratio))
        ratio = ratio ** self.curve
        mapped = self.min_speed + ratio * (self.max_speed - self.min_speed)
        self.current_speed = int(round(mapped))
        self.prev_center = center
        self.prev_time = timestamp
        self.anchor_center = center
        self.anchor_time = timestamp
        return self.current_speed, self.smoothed_velocity

    def update(self, box, timestamp: float) -> tuple[int, float]:
        return self.update_center(get_box_center(box), timestamp)

    def peek(self) -> tuple[int, float]:
        return self.current_speed, self.smoothed_velocity


class HandOpticalFlowTracker:
    """Track hand-box center cheaply between slower detector frames."""

    def __init__(self, max_points: int = 40, min_points: int = 6):
        self.max_points = max_points
        self.min_points = min_points
        self.prev_gray: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None
        self.box_norm: np.ndarray | None = None

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.box_norm = None

    def seed(self, frame_bgr: np.ndarray, box) -> np.ndarray:
        self.reset()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = gray.shape[:2]
        self.box_norm = np.asarray(box.detach().cpu().numpy(), dtype=np.float32).copy()
        x1, y1, x2, y2 = self._box_to_pixels(self.box_norm, frame_w, frame_h)

        roi = gray[y1:y2, x1:x2]
        points = None
        if roi.size > 0 and roi.shape[0] >= 4 and roi.shape[1] >= 4:
            points = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=self.max_points,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=5,
            )

        if points is None or len(points) < self.min_points:
            points = self._fallback_points(x1, y1, x2, y2)
        else:
            points[:, 0, 0] += x1
            points[:, 0, 1] += y1

        self.prev_gray = gray
        self.prev_points = points.astype(np.float32)
        return get_box_center_from_array(self.box_norm)

    def update(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if self.prev_gray is None or self.prev_points is None or self.box_norm is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            self.reset()
            return None

        valid = status.reshape(-1) == 1
        if int(valid.sum()) < self.min_points:
            self.reset()
            return None

        old_valid = self.prev_points[valid].reshape(-1, 2)
        new_valid = next_points[valid].reshape(-1, 2)
        shift_px = np.median(new_valid - old_valid, axis=0)

        frame_h, frame_w = gray.shape[:2]
        dx = float(shift_px[0] / max(1, frame_w))
        dy = float(shift_px[1] / max(1, frame_h))
        self.box_norm[[0, 2]] += dx
        self.box_norm[[1, 3]] += dy
        self.box_norm = self._clip_box(self.box_norm)

        self.prev_gray = gray
        self.prev_points = next_points[valid].reshape(-1, 1, 2).astype(np.float32)
        return self.box_norm.copy(), get_box_center_from_array(self.box_norm)

    @staticmethod
    def _clip_box(box: np.ndarray) -> np.ndarray:
        width = max(1e-4, float(box[2] - box[0]))
        height = max(1e-4, float(box[3] - box[1]))
        box[0] = max(0.0, min(1.0 - width, float(box[0])))
        box[1] = max(0.0, min(1.0 - height, float(box[1])))
        box[2] = min(1.0, box[0] + width)
        box[3] = min(1.0, box[1] + height)
        return box

    @staticmethod
    def _box_to_pixels(box: np.ndarray, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(frame_w - 2, int(round(x1 * frame_w))))
        y1 = max(0, min(frame_h - 2, int(round(y1 * frame_h))))
        x2 = max(x1 + 1, min(frame_w - 1, int(round(x2 * frame_w))))
        y2 = max(y1 + 1, min(frame_h - 1, int(round(y2 * frame_h))))
        return x1, y1, x2, y2

    @staticmethod
    def _fallback_points(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        points = np.asarray(
            [
                [cx, cy],
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2],
                [(x1 + cx) * 0.5, cy],
                [(x2 + cx) * 0.5, cy],
                [cx, (y1 + cy) * 0.5],
                [cx, (y2 + cy) * 0.5],
            ],
            dtype=np.float32,
        )
        return points.reshape(-1, 1, 2)


def make_single_prediction(box_norm: np.ndarray, score: float, label: int):
    return {
        "boxes": torch.as_tensor([box_norm], dtype=torch.float32),
        "scores": torch.as_tensor([score], dtype=torch.float32),
        "labels": torch.as_tensor([label], dtype=torch.long),
    }


def send_car_command(car_ip: str, command: str, speed: int, timeout: float = 1.0):
    try:
        if command == "stop":
            speed_l = 0
            speed_r = 0
        else:
            active_speed = int(speed * TURN_RATIO) if command in {"left", "right"} else speed
            speed_l = int(active_speed * LEFT_TRIM)
            speed_r = int(active_speed * RIGHT_TRIM)

        response = requests.get(
            f"{car_ip}/{command}",
            params={"speedL": speed_l, "speedR": speed_r},
            timeout=timeout,
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


class CameraStream:
    def __init__(self, camera_index: int, width: int | None = None, height: int | None = None):
        self.cap = open_camera(camera_index)
        if width is not None and width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None and height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


class CommandSender:
    def __init__(self, car_ip: str):
        self.car_ip = car_ip
        self.queue: Queue[tuple[str, int]] = Queue(maxsize=1)
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.last_status = False
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def enqueue(self, command: str, speed: int):
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass
        self.queue.put_nowait((command, speed))

    def _worker(self):
        while not self.stopped.is_set() or not self.queue.empty():
            try:
                command, speed = self.queue.get(timeout=0.1)
            except Empty:
                continue
            status = send_car_command(self.car_ip, command, speed)
            with self.lock:
                self.last_status = status
            print(f"[CommandSender] sent command={command} speed={speed} status={status}", flush=True)

    def get_last_status(self) -> bool:
        with self.lock:
            return self.last_status

    def stop(self):
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def draw_status_panel(
    frame_bgr: np.ndarray,
    command: str,
    gesture: str,
    speed: int,
    locked_command: str,
    locked_speed: int,
    car_ip: str,
    sent: bool,
    confidence: float,
    box_area: float,
    motion_velocity: float,
    speed_mode: str,
):
    panel_color = (30, 30, 30)
    overlay = frame_bgr.copy()
    frame_h, frame_w = frame_bgr.shape[:2]
    panel_right = min(frame_w - 10, 500)
    cv2.rectangle(overlay, (10, 10), (panel_right, 215), panel_color, thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    status_color = (0, 200, 0) if sent else (0, 0, 255)
    cv2.putText(frame_bgr, f"Gesture: {gesture}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Command: {command}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Speed: {speed}", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Locked: {locked_command}-{locked_speed}", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Mode: {speed_mode}", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Motion: {motion_velocity:.2f}", (25, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Box: {box_area:.3f}", (235, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Car: {car_ip}", (225, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(frame_bgr, (404, 108), 10, status_color, thickness=-1)
    cv2.putText(frame_bgr, "SENT" if sent else "NO LINK", (360, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame_bgr


def open_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return cap


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time hand gesture detection from webcam.")
    parser.add_argument("--model", choices=["resnet", "mobilenet", "yolo"], default=None, help="Which trained model to use (if not specified, will prompt interactively)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to the .pt model file (optional)")
    parser.add_argument("--mobilenet-checkpoint", type=Path, default=DEFAULT_MOBILENET_CHECKPOINT, help="MobileNet SSD checkpoint path")
    parser.add_argument("--yolo-checkpoint", type=Path, default=None, help="YOLO checkpoint path (ultralytics)")
    parser.add_argument("--yolo-variant", choices=["yolo26", "yolo11n"], default="yolo11n", help="Which pretrained YOLO variant to use when --model=yolo")
    parser.add_argument("--yolo-img-size", type=int, default=416, help="Realtime YOLO inference image size; lower is faster")
    parser.add_argument("--yolo-max-det", type=int, default=1, help="Maximum YOLO detections to keep per frame")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--camera-width", type=int, default=640, help="Webcam capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Webcam capture height")
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.75,
        help="Scale factor for the preview window; lower values improve UI FPS",
    )
    parser.add_argument("--score-threshold", type=float, default=0.30, help="Objectness threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument(
        "--detection-mode",
        choices=["single", "multi"],
        default="single",
        help="Detection mode for grid models: single keeps only top-1, multi allows multiple boxes",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=3,
        help="Maximum detections to keep in multi mode",
    )
    parser.add_argument(
        "--no-hand-frames",
        type=int,
        default=3,
        help="Consecutive missed inference frames required before declaring no hand",
    )
    parser.add_argument(
        "--hold-last-detection-ms",
        type=int,
        default=180,
        help="How long to keep the last valid detection during brief dropouts",
    )
    parser.add_argument(
        "--inference-stride",
        type=int,
        default=4,
        help="Run the neural detector every N displayed frames before any active-command cooldown",
    )
    parser.add_argument(
        "--detect-cooldown-ms",
        type=int,
        default=0,
        help="After a valid active gesture is detected, wait this long before running neural detection again",
    )
    parser.add_argument(
        "--speed-mode",
        choices=["motion", "box"],
        default="motion",
        help="motion uses hand center speed; box keeps the old hand-size speed mapping",
    )
    parser.add_argument(
        "--motion-low",
        type=float,
        default=DEFAULT_MOTION_LOW,
        help="Normalized hand velocity that maps to the minimum active car speed",
    )
    parser.add_argument(
        "--motion-high",
        type=float,
        default=DEFAULT_MOTION_HIGH,
        help="Normalized hand velocity that maps to the maximum active car speed",
    )
    parser.add_argument(
        "--motion-deadzone",
        type=float,
        default=DEFAULT_MOTION_DEADZONE,
        help="Ignore normalized hand velocities below this value as detector jitter",
    )
    parser.add_argument(
        "--motion-smoothing",
        type=float,
        default=DEFAULT_MOTION_SMOOTHING,
        help="EMA factor for motion speed smoothing; larger values react faster",
    )
    parser.add_argument(
        "--motion-curve",
        type=float,
        default=DEFAULT_MOTION_CURVE,
        help="Motion response curve; lower values make slow movement affect speed more",
    )
    parser.add_argument(
        "--command-window",
        type=int,
        default=3,
        help="Number of speed samples to average before sending a car command",
    )
    parser.add_argument(
        "--repeat-command-ms",
        type=int,
        default=120,
        help="Repeat the last stable active command at this interval so slow inference does not pause the car",
    )
    parser.add_argument(
        "--speed-change-threshold",
        type=int,
        default=12,
        help="Minimum speed change required before updating a latched command speed",
    )
    parser.add_argument("--car-ip", type=str, default=DEFAULT_CAR_IP, help="Base URL of the wireless car controller")
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam preview")
    return parser.parse_args()


def interactive_model_selection():
    """Prompt user to choose a model interactively.

    Returns a tuple: (model_choice, yolo_variant_or_None).
    """
    print("\n" + "="*50)
    print("SELECT A MODEL FOR HAND DETECTION")
    print("="*50)
    print("1) ResNet18 (default, balanced speed/accuracy)")
    print("2) MobileNet SSD (faster, mobile-optimized)")
    print("3) YOLO (accurate but requires ultralytics)")
    print("="*50)

    while True:
        choice = input("Enter choice (1-3) [default: 1]: ").strip()
        if choice == "" or choice == "1":
            return "resnet", None
        elif choice == "2":
            return "mobilenet", None
        elif choice == "3":
            # Prompt for YOLO variant when user picks YOLO in the UI
            print("\nChoose YOLO variant:")
            print("1) yolo11n (default, fastest)")
            print("2) yolo26")
            while True:
                v = input("Enter choice (1-2) [default: 1]: ").strip()
                if v == "" or v == "1":
                    return "yolo", "yolo11n"
                elif v == "2":
                    return "yolo", "yolo26"
                else:
                    print("Invalid choice. Please enter 1 or 2.")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select checkpoint path based on requested model
    if args.model is None:
        chosen, interactive_variant = interactive_model_selection()
        # If user selected a YOLO variant in the interactive UI, honor it
        if interactive_variant is not None:
            args.yolo_variant = interactive_variant
    else:
        chosen = args.model
        interactive_variant = None

    print(f"\nSelected model: {chosen}")

    ckpt = None  # Track the actual checkpoint used

    grid_model = None
    yolo_model = None

    if chosen == "resnet":
        ckpt = args.checkpoint or DEFAULT_CHECKPOINT
        if not ckpt.exists():
            raise FileNotFoundError(f"ResNet checkpoint not found: {ckpt}")
        grid_model, id_to_class, img_size, grid_size = load_checkpoint(ckpt, device)
        model_kind = "grid"
    elif chosen == "mobilenet":
        ckpt = args.checkpoint or args.mobilenet_checkpoint
        if not ckpt.exists():
            raise FileNotFoundError(f"MobileNet checkpoint not found: {ckpt}")
        grid_model, id_to_class, img_size, grid_size = load_mobilenet_checkpoint(ckpt, device)
        model_kind = "grid"
    elif chosen == "yolo":
        # Pick explicit checkpoint priority: --checkpoint > --yolo-checkpoint > variant default
        ckpt = args.checkpoint or args.yolo_checkpoint
        if ckpt is None:
            ckpt = DEFAULT_YOLO11N_CHECKPOINT if args.yolo_variant == "yolo11n" else DEFAULT_YOLO26_CHECKPOINT
        if not ckpt.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt}")
        yolo_model, id_to_class, img_size, grid_size = load_yolo_model(
            ckpt,
            device,
            img_size=args.yolo_img_size,
            max_det=args.yolo_max_det,
        )
        yolo_model.iou_threshold = args.iou_threshold
        model_kind = "yolo"
    else:
        raise ValueError(f"Unknown model choice: {chosen}")

    last_command = None
    last_speed = None
    last_send_time = 0.0
    last_detection = None
    next_detection_time = 0.0
    last_valid_detection_time = 0.0
    no_hand_streak = 0
    last_gesture = "none"
    last_box_area = 0.0
    last_confidence = 0.0
    frame_index = 0
    inference_stride = max(1, args.inference_stride)
    speed_buffer: list[int] = []
    pending_command = "stop"
    last_label = 0
    motion_tracker = HandMotionSpeedTracker(
        min_speed=MIN_CAR_SPEED,
        max_speed=MAX_CAR_SPEED,
        low_velocity=args.motion_low,
        high_velocity=args.motion_high,
        deadzone=args.motion_deadzone,
        smoothing=args.motion_smoothing,
        curve=args.motion_curve,
    )

    camera_stream = CameraStream(args.camera, args.camera_width, args.camera_height).start()
    sender = CommandSender(args.car_ip).start()
    last_fps_time = time.perf_counter()
    fps = 0.0
    frame_times = []  # Rolling window for FPS calculation
    fps_window_size = 30  # Average over 30 frames for smooth FPS

    print(f"Loaded checkpoint: {ckpt}")
    print(f"Model type: {model_kind}")
    print(f"Device: {device}")
    print("Press Q or ESC to quit.")

    try:
        with torch.inference_mode():
            while True:
                frame = camera_stream.read()
                if frame is None:
                    time.sleep(0.005)
                    continue

                if args.mirror:
                    frame = cv2.flip(frame, 1)

                loop_now = time.perf_counter()
                frame_h, frame_w = frame.shape[:2]
                gesture = last_gesture
                command = COMMAND_MAP.get(gesture, "stop") if gesture != "none" else "stop"
                detection_due = loop_now >= next_detection_time
                should_infer = last_detection is None or (
                    detection_due and (frame_index % inference_stride) == 0
                )
                speed = 0 if command == "stop" else (last_speed if last_speed is not None else MIN_CAR_SPEED)
                confidence = last_confidence
                box_area = last_box_area
                motion_velocity = motion_tracker.peek()[1]
                sent = sender.get_last_status()
                predictions = last_detection

                if should_infer:
                    raw_predictions = None
                    if model_kind == "grid":
                        assert grid_model is not None
                        input_tensor = preprocess_frame(frame, img_size, device)
                        box_preds, cls_logits = grid_model(input_tensor)
                        probs = torch.softmax(cls_logits, dim=-1)
                        scores, labels = torch.max(probs, dim=1)
                        x_c, y_c, w, h = box_preds.unbind(1)
                        pred_boxes_xyxy = torch.stack([x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2], dim=1).clamp(0, 1)
                        raw_predictions = {
                            "boxes": pred_boxes_xyxy,
                            "scores": scores,
                            "labels": labels
                        }
                    else:
                        assert yolo_model is not None
                        raw_predictions = yolo_model.predict_frame(frame, score_threshold=0.5)

                    # Enforce single highest confidence prediction >= 0.5
                    if raw_predictions is not None and len(raw_predictions["boxes"]) > 0:
                        boxes = raw_predictions["boxes"]
                        scores = raw_predictions["scores"]
                        labels = raw_predictions["labels"]
                        
                        keep_mask = scores >= 0.5
                        if not keep_mask.any():
                            raw_predictions = None
                        else:
                            boxes = boxes[keep_mask]
                            scores = scores[keep_mask]
                            labels = labels[keep_mask]
                            
                            best_idx = torch.argmax(scores)
                            raw_predictions = {
                                "boxes": boxes[best_idx].unsqueeze(0),
                                "scores": scores[best_idx].unsqueeze(0),
                                "labels": labels[best_idx].unsqueeze(0)
                            }
                    
                    now = time.perf_counter()
                    hold_seconds = max(0.0, float(args.hold_last_detection_ms) / 1000.0)
                    has_valid = raw_predictions is not None and len(raw_predictions["boxes"]) > 0

                    if has_valid:
                        no_hand_streak = 0
                        last_detection = raw_predictions
                        last_valid_detection_time = now
                        predictions = raw_predictions
                    else:
                        no_hand_streak += 1
                        can_hold_last = (
                            last_detection is not None
                            and (now - last_valid_detection_time) <= hold_seconds
                            and no_hand_streak < args.no_hand_frames
                        )
                        predictions = last_detection if can_hold_last else None

                    if predictions is not None and len(predictions["boxes"]) > 0:
                        box = predictions["boxes"][0]
                        score = float(predictions["scores"][0].item())
                        label = int(predictions["labels"][0].item())
                        gesture = id_to_class.get(label, str(label))
                        command = COMMAND_MAP.get(gesture, "stop")
                        x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
                        box_w = max(0.0, x2 - x1)
                        box_h = max(0.0, y2 - y1)
                        # Print the exact mathematical output format required by the problem statement
                        print(f"y_i = ({score:.3f}, [{x1:.3f}, {y1:.3f}, {box_w:.3f}, {box_h:.3f}], {label + 1})  # {gesture}")
                        box_area = max(0.0, min(1.0, box_w * box_h))
                        if command == "stop":
                            speed = 0
                            motion_tracker.reset()
                            next_detection_time = 0.0
                            motion_velocity = 0.0
                        elif args.speed_mode == "motion":
                            if has_valid:
                                center = get_box_center(box)
                                speed, motion_velocity = motion_tracker.update_center(center, now)
                                cooldown_seconds = max(0.0, float(args.detect_cooldown_ms) / 1000.0)
                                next_detection_time = now + cooldown_seconds
                            else:
                                speed, motion_velocity = motion_tracker.peek()
                        else:
                            speed = compute_speed_from_box_size(box)
                            # Apply per-class size weight only to the legacy
                            # box-size mode because it compensates for size bias.
                            weight = CLASS_SIZE_WEIGHTS.get(gesture, 1.0)
                            speed = int(round(speed * weight))
                            speed = max(MIN_CAR_SPEED, min(MAX_CAR_SPEED, speed))
                            cooldown_seconds = max(0.0, float(args.detect_cooldown_ms) / 1000.0)
                            next_detection_time = now + cooldown_seconds
                        confidence = score
                        last_gesture = gesture
                        last_label = label
                        last_box_area = box_area
                        last_confidence = confidence
                    else:
                        gesture = "none"
                        command = "stop"
                        speed = 0
                        confidence = 0.0
                        box_area = 0.0
                        motion_velocity = 0.0
                        last_detection = None
                        last_gesture = gesture
                        last_box_area = box_area
                        last_confidence = confidence
                        motion_tracker.reset()
                        next_detection_time = 0.0

                now = time.perf_counter()
                repeat_interval = max(0.02, float(args.repeat_command_ms) / 1000.0)
                speed_change_threshold = max(0, int(args.speed_change_threshold))
                command_window = max(1, int(args.command_window))

                if gesture == "none" or command == "stop":
                    speed = 0
                    speed_buffer.clear()
                    if command != last_command or last_speed != 0 or (now - last_send_time) > 0.1:
                        sender.enqueue("stop", 0)
                        last_command = "stop"
                        last_speed = 0
                        last_send_time = now
                else:
                    if command != pending_command:
                        pending_command = command
                        speed_buffer.clear()

                    speed = max(MIN_CAR_SPEED, int(speed))
                    speed_buffer.append(speed)
                    if len(speed_buffer) > command_window:
                        speed_buffer.pop(0)

                    averaged_speed = int(round(sum(speed_buffer) / len(speed_buffer)))
                    has_meaningful_speed_change = (
                        last_speed is None
                        or abs(averaged_speed - int(last_speed)) >= speed_change_threshold
                    )
                    should_repeat_locked_command = (now - last_send_time) >= repeat_interval

                    if command != last_command or has_meaningful_speed_change or should_repeat_locked_command:
                        sender.enqueue(command, averaged_speed)
                        last_command = command
                        last_speed = averaged_speed
                        last_send_time = now

                display_scale = max(0.2, min(1.0, float(args.display_scale)))
                if display_scale < 0.999:
                    display_frame = cv2.resize(
                        frame,
                        None,
                        fx=display_scale,
                        fy=display_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    display_frame = frame.copy()
                display_h, display_w = display_frame.shape[:2]

                display_frame = draw_predictions(display_frame, predictions, id_to_class, display_w, display_h)
                display_frame = draw_status_panel(
                    display_frame,
                    command,
                    gesture,
                    speed,
                    last_command or "none",
                    int(last_speed or 0),
                    args.car_ip,
                    sent,
                    confidence,
                    box_area,
                    motion_velocity,
                    args.speed_mode,
                )

                now = time.perf_counter()
                elapsed = now - last_fps_time
                frame_times.append(elapsed)
                if len(frame_times) > fps_window_size:
                    frame_times.pop(0)
                if len(frame_times) > 1:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    if avg_frame_time > 0:
                        fps = 1.0 / avg_frame_time
                last_fps_time = now
                frame_index += 1

                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (20, display_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Real-time Hand Detector", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        sender.enqueue("stop", 0)
        sender.stop()
        camera_stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
