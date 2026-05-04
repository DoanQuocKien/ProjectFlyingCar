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
DEFAULT_CAR_IP = "http://192.168.137.228"
COMMAND_MAP = {
    "one": "forward",
    "peace": "right",
    "three": "left",
    "four": "backward",
    "fist": "stop",
}

# Per-class size bias (optional tuning)
CLASS_SIZE_WEIGHTS = {
    "one": 1.0,
    "peace": 1.0,
    "three": 1.0,
    "four": 0.85,
    "fist": 1.0,
}


class ResNetDetector(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.detection_head = nn.Conv2d(512, 1 + 4 + num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        out = self.detection_head(feat)
        out = out.permute(0, 2, 3, 1)
        obj_logits = out[..., 0]
        box_preds = torch.sigmoid(out[..., 1:5])
        cls_logits = out[..., 5:]
        return obj_logits, box_preds, cls_logits


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")

    class_to_id = checkpoint.get("class_to_id") or {name: idx for idx, name in enumerate(DEFAULT_CLASSES)}
    id_to_class = {idx: name for name, idx in class_to_id.items()}
    num_classes = len(class_to_id)
    img_size = int(checkpoint.get("img_size", 448))
    grid_size = int(checkpoint.get("grid_size", 14))

    model = ResNetDetector(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, id_to_class, img_size, grid_size


# --- MobileNet SSD transfer model (from notebook) ---
class TransferMobileNetSSD(nn.Module):
    """Pretrained MobileNetV3-Large backbone + SSD-style grid head."""
    def __init__(self, num_classes: int = 5, grid_size: int = 10, freeze_backbone_until: int = 13):
        super().__init__()
        # Use torchvision's mobilenet_v3_large features as backbone
        mobilenet = models.mobilenet_v3_large(weights=None)
        self.backbone = mobilenet.features
        self.grid_size = grid_size
        self.num_classes = num_classes

        # Adapter conv: MobileNetV3-Large outputs 960 channels
        # Use Hardswish to match MobileNetV3 internals and preserve activation statistics.
        self.adapt_conv = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True)
        )

        # SSD-like detection head (keeps same channel layout expected by loss)
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, 1 + 4 + num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        feat = self.adapt_conv(feat)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, output_size=(self.grid_size, self.grid_size))

        out = self.detection_head(feat)
        out = out.permute(0, 2, 3, 1)
        obj_logits = out[..., 0]
        box_preds = torch.sigmoid(out[..., 1:5])
        cls_logits = out[..., 5:]
        return obj_logits, box_preds, cls_logits


def load_mobilenet_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")

    class_to_id = checkpoint.get("class_to_id") or {name: idx for idx, name in enumerate(DEFAULT_CLASSES)}
    id_to_class = {idx: name for name, idx in class_to_id.items()}
    num_classes = len(class_to_id)
    img_size = int(checkpoint.get("img_size", 320))
    grid_size = int(checkpoint.get("grid_size", 10))

    model = TransferMobileNetSSD(num_classes=num_classes, grid_size=grid_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, id_to_class, img_size, grid_size


# --- YOLO wrapper (ultralytics) ---
class YOLOWrapper:
    def __init__(self, checkpoint_path: Path, device: torch.device, img_size: int = 640):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics YOLO is required for --model=yolo") from exc
        self.model = YOLO(str(checkpoint_path))
        self.device = device
        self.img_size = img_size

    def predict_frame(self, frame_bgr: np.ndarray, score_threshold: float = 0.25):
        # ultralytics accepts RGB numpy arrays
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb, imgsz=self.img_size, conf=score_threshold)
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
        h, w = frame_bgr.shape[:2]
        boxes_norm = torch.from_numpy(xyxy / np.array([w, h, w, h], dtype=float)).float()
        scores = torch.from_numpy(confs).float()
        labels = torch.from_numpy(cls).long()
        return {"boxes": boxes_norm, "scores": scores, "labels": labels}


def load_yolo_model(checkpoint_path: Path, device: torch.device):
    # Return a wrapper and dummy id_to_class mapping
    wrapper = YOLOWrapper(checkpoint_path, device)
    id_to_class = {i: name for i, name in enumerate(DEFAULT_CLASSES)}
    return wrapper, id_to_class, wrapper.img_size, None


def decode_predictions(
    obj_logits: torch.Tensor,
    box_preds: torch.Tensor,
    cls_logits: torch.Tensor,
    grid_size: int,
    score_threshold: float,
    iou_threshold: float,
):
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

        batch_predictions.append(
            {
                "boxes": boxes[kept_indices],
                "scores": filtered_probs[kept_indices],
                "labels": class_ids[kept_indices],
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
    min_speed: int = 127,
    max_speed: int = 255,
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


def send_car_command(car_ip: str, command: str, speed: int, timeout: float = 1.0):
    try:
        response = requests.get(f"{car_ip}/{command}", params={"speed": speed}, timeout=timeout)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


class CameraStream:
    def __init__(self, camera_index: int):
        self.cap = open_camera(camera_index)
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
    car_ip: str,
    sent: bool,
    confidence: float,
    box_area: float,
):
    panel_color = (30, 30, 30)
    overlay = frame_bgr.copy()
    frame_h, frame_w = frame_bgr.shape[:2]
    panel_right = min(frame_w - 10, 430)
    cv2.rectangle(overlay, (10, 10), (panel_right, 155), panel_color, thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0, frame_bgr)

    status_color = (0, 200, 0) if sent else (0, 0, 255)
    cv2.putText(frame_bgr, f"Gesture: {gesture}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Command: {command}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Speed: {speed}", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Box: {box_area:.3f}", (25, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
    parser.add_argument("--yolo-checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT, help="YOLO checkpoint path (ultralytics)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--score-threshold", type=float, default=0.30, help="Objectness threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument("--car-ip", type=str, default=DEFAULT_CAR_IP, help="Base URL of the wireless car controller")
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam preview")
    return parser.parse_args()


def interactive_model_selection():
    """Prompt user to choose a model interactively."""
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
            return "resnet"
        elif choice == "2":
            return "mobilenet"
        elif choice == "3":
            return "yolo"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select checkpoint path based on requested model
    chosen = args.model or interactive_model_selection()
    print(f"\nSelected model: {chosen}")

    ckpt = None  # Track the actual checkpoint used

    if chosen == "resnet":
        ckpt = args.checkpoint or DEFAULT_CHECKPOINT
        if not ckpt.exists():
            raise FileNotFoundError(f"ResNet checkpoint not found: {ckpt}")
        model, id_to_class, img_size, grid_size = load_checkpoint(ckpt, device)
        model_kind = "grid"
    elif chosen == "mobilenet":
        ckpt = args.checkpoint or args.mobilenet_checkpoint
        if not ckpt.exists():
            raise FileNotFoundError(f"MobileNet checkpoint not found: {ckpt}")
        model, id_to_class, img_size, grid_size = load_mobilenet_checkpoint(ckpt, device)
        model_kind = "grid"
    elif chosen == "yolo":
        ckpt = args.checkpoint or args.yolo_checkpoint
        if not ckpt.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt}")
        yolo_wrapper, id_to_class, img_size, grid_size = load_yolo_model(ckpt, device)
        model = yolo_wrapper
        model_kind = "yolo"
    else:
        raise ValueError(f"Unknown model choice: {chosen}")

    last_command = None
    last_speed = None
    last_send_time = 0.0
    last_detection = None
    last_gesture = "none"
    last_box_area = 0.0
    last_confidence = 0.0
    frame_index = 0
    inference_stride = 2
    speed_buffer: list[int] = []
    pending_command = "stop"

    camera_stream = CameraStream(args.camera).start()
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

                frame_h, frame_w = frame.shape[:2]
                should_infer = (frame_index % inference_stride) == 0 or last_detection is None

                gesture = last_gesture
                command = COMMAND_MAP.get(gesture, "stop") if gesture != "none" else "stop"
                speed = 0 if command == "stop" else (last_speed if last_speed is not None else 127)
                confidence = last_confidence
                box_area = last_box_area
                sent = sender.get_last_status()

                if should_infer:
                    if model_kind == "grid":
                        input_tensor = preprocess_frame(frame, img_size, device)
                        obj_logits, box_preds, cls_logits = model(input_tensor)
                        predictions = decode_predictions(
                            obj_logits,
                            box_preds,
                            cls_logits,
                            grid_size=grid_size,
                            score_threshold=args.score_threshold,
                            iou_threshold=args.iou_threshold,
                        )[0]
                    else:
                        # YOLO pipeline returns normalized boxes directly
                        predictions = model.predict_frame(frame, score_threshold=args.score_threshold)

                    last_detection = predictions

                    if predictions is not None and len(predictions["boxes"]) > 0:
                        box = predictions["boxes"][0]
                        score = float(predictions["scores"][0].item())
                        label = int(predictions["labels"][0].item())
                        gesture = id_to_class.get(label, str(label))
                        command = COMMAND_MAP.get(gesture, "stop")
                        x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
                        box_area = max(0.0, min(1.0, max(0.0, x2 - x1) * max(0.0, y2 - y1)))
                        speed = compute_speed_from_box_size(box)
                        # Apply per-class size weight to compensate for
                        # gestures that tend to appear larger.
                        weight = CLASS_SIZE_WEIGHTS.get(gesture, 1.0)
                        speed = int(round(speed * weight))
                        # Clamp to requested range for active commands
                        speed = max(127, min(255, speed))
                        confidence = score
                        last_gesture = gesture
                        last_box_area = box_area
                        last_confidence = confidence
                    else:
                        gesture = "none"
                        command = "stop"
                        speed = 0
                        confidence = 0.0
                        box_area = 0.0
                        last_gesture = gesture
                        last_box_area = box_area
                        last_confidence = confidence

                    frame = draw_predictions(frame, predictions, id_to_class, frame_w, frame_h)
                else:
                    frame = draw_predictions(frame, last_detection, id_to_class, frame_w, frame_h)

                now = time.perf_counter()
                if gesture == "none":
                    speed = 0
                    speed_buffer.clear()
                    if command != last_command or last_speed != 0 or (now - last_send_time) > 0.1:
                        sender.enqueue("stop", 0)
                        last_command = "stop"
                        last_speed = 0
                        last_send_time = now
                else:
                    if command != pending_command and speed_buffer:
                        averaged_speed = int(round(sum(speed_buffer) / len(speed_buffer)))
                        sender.enqueue(pending_command, averaged_speed)
                        last_command = pending_command
                        last_speed = averaged_speed
                        last_send_time = now
                        speed_buffer.clear()

                    if command != pending_command:
                        pending_command = command

                    speed_buffer.append(speed)
                    if len(speed_buffer) >= 5:
                        averaged_speed = int(round(sum(speed_buffer) / len(speed_buffer)))
                        sender.enqueue(command, averaged_speed)
                        last_command = command
                        last_speed = averaged_speed
                        last_send_time = now
                        speed_buffer.clear()

                frame = draw_status_panel(
                    frame,
                    command,
                    gesture,
                    speed,
                    args.car_ip,
                    sent,
                    confidence,
                    box_area,
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
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, frame_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Real-time Hand Detector", frame)
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