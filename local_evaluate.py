import os
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import matplotlib.pyplot as plt

# -----------------
# 1. Dataset Logic
# -----------------
TARGET_CLASSES = ["one", "peace", "three", "four", "fist"]
CLASS_TO_ID = {c: i for i, c in enumerate(TARGET_CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_filename_index(root: Path):
    filename_index = {}
    for img_path in root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
            filename_index.setdefault(img_path.name, img_path)
            filename_index.setdefault(img_path.stem, img_path)
    return filename_index

def resolve_image_path(filename_index, image_key):
    key = Path(image_key).stem
    return filename_index.get(key) or filename_index.get(Path(image_key).name)

def xywh_to_xyxy(box, w, h, normalized=True):
    x, y, bw, bh = box
    if normalized:
        x *= w; y *= h; bw *= w; bh *= h
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(w - 1), x + bw), min(float(h - 1), y + bh)
    return [x1, y1, x2, y2]

def parse_hagrid_annotations(dataset_root: Path):
    ann_dir = dataset_root / "ann_train_val"
    img_root = dataset_root / "hagrid_30k"
    
    json_files = [ann_dir / f"{c}.json" for c in TARGET_CLASSES if (ann_dir / f"{c}.json").exists()]
    if not json_files: return []
    
    filename_index = build_filename_index(img_root)
    records = []
    
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        for image_key, item in data.items():
            labels, bboxes = item.get("labels", []), item.get("bboxes", [])
            if not labels or not bboxes: continue
            
            img_path = resolve_image_path(filename_index, image_key)
            if img_path is None: continue
            
            try:
                with Image.open(img_path) as img_pil:
                    w, h = img_pil.size
            except Exception: continue
            
            for label, box in zip(labels, bboxes):
                if label not in TARGET_CLASSES or len(box) < 4: continue
                normalized = max(box[:4]) <= 1.5
                x1, y1, x2, y2 = xywh_to_xyxy(box[:4], w, h, normalized=normalized)
                records.append({
                    "image_path": str(img_path),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "class_id": CLASS_TO_ID[label],
                })
    
    # Filter single-hand
    img_counts = Counter(r['image_path'] for r in records)
    return [r for r in records if img_counts[r['image_path']] == 1]

class GestureLocalizationDataset(Dataset):
    def __init__(self, records, img_size=384):
        self.records = records
        self.img_size = img_size
        
    def __len__(self): return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = record["image_path"]
        with Image.open(img_path) as img_pil:
            img_pil = img_pil.convert("RGB")
            w, h = img_pil.size
            img_t = TF.to_tensor(img_pil)
        
        img_t = TF.resize(img_t, [self.img_size, self.img_size], antialias=True)
        
        x1, y1, x2, y2 = record["bbox_xyxy"]
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        
        return {
            "image": img_t,
            "target_box": torch.tensor([xc, yc, bw, bh], dtype=torch.float32),
            "target_label": torch.tensor(record["class_id"], dtype=torch.long),
            "path": img_path
        }

# -----------------
# 2. Models (UPDATED DEEP MLP ARCHITECTURE)
# -----------------
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
        feat = self.pool(self.backbone(x)).flatten(1)
        out = self.head(feat)
        return torch.sigmoid(out[:, :4]), out[:, 4:]

class ResNetLocalization(nn.Module):
    def __init__(self, num_classes=5):
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
    def forward(self, x):
        feat = self.pool(self.backbone(x)).flatten(1)
        out = self.head(feat)
        return torch.sigmoid(out[:, :4]), out[:, 4:]

def decode_batch_predictions(box_preds, cls_logits):
    pred_labels = torch.argmax(cls_logits, dim=1).cpu().tolist()
    pred_scores = torch.softmax(cls_logits, dim=-1).max(dim=1)[0].cpu().tolist()
    xc, yc, w_b, h_b = box_preds.unbind(1)
    pred_boxes_xyxy = torch.stack([xc - w_b/2, yc - h_b/2, xc + w_b/2, yc + h_b/2], dim=1).clamp(0, 1)
    return pred_labels, pred_scores, pred_boxes_xyxy

# -----------------
# 3. Evaluation
# -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "mobilenet"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="Limit test samples")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    data_root = Path("data/hagrid-sample-30k-384p-5class")
    print(f"Loading local dataset from {data_root}...")
    records = parse_hagrid_annotations(data_root)
    if not records:
        print("No records found. Check dataset path.")
        return
        
    np.random.seed(42)
    np.random.shuffle(records)
    n_test = int(len(records) * 0.15)
    test_records = records[-n_test:]
    if args.limit > 0:
        test_records = test_records[:args.limit]
        
    test_dataset = GestureLocalizationDataset(test_records)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_records)}")
    
    # Load Model
    if args.model == "mobilenet":
        model = MobileNetLocalization(num_classes=5).to(device)
        pt_path = Path("models/mobilenet_ssd/mobilenet_ssd_hagrid_detector.pt")
        out_dir = Path("models/mobilenet_ssd")
    else:
        model = ResNetLocalization(num_classes=5).to(device)
        pt_path = Path("models/resnet18/resnet18_hagrid_detector.pt")
        out_dir = Path("models/resnet18")
        
    if not pt_path.exists():
        print(f"Model not found at {pt_path}! Please ensure the downloaded .pt file is placed there.")
        return
        
    print(f"Loading {pt_path}...")
    checkpoint = torch.load(pt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    print(f"Evaluating {args.model} on Local Test Set...")
    metric = MeanAveragePrecision(iou_type="bbox").to(device)
    ious, latencies = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x = batch["image"].to(device)
            target_boxes = batch["target_box"].to(device)
            target_labels = batch["target_label"].to(device)
            
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            box_preds, cls_logits = model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            latencies.extend([(t1 - t0) * 1000.0 / x.size(0)] * x.size(0))
            
            p_l, p_s, p_b = decode_batch_predictions(box_preds, cls_logits)
            g_xc, g_yc, g_w, g_h = target_boxes.unbind(1)
            g_b = torch.stack([g_xc - g_w/2, g_yc - g_h/2, g_xc + g_w/2, g_yc + g_h/2], dim=1).clamp(0, 1)
            g_l = target_labels.cpu().tolist()
            
            ious.extend(torch.diag(box_iou(p_b.cpu(), g_b.cpu())).numpy().tolist())
            
            preds, target = [], []
            for j in range(len(p_l)):
                preds.append({"boxes": p_b[j].unsqueeze(0).to(device) * 384.0, "scores": torch.tensor([p_s[j]], device=device), "labels": torch.tensor([p_l[j]], device=device)})
                target.append({"boxes": g_b[j].unsqueeze(0).to(device) * 384.0, "labels": torch.tensor([g_l[j]], device=device)})
            metric.update(preds, target)
            
    mAP_dict = metric.compute()
    print("\n" + "="*40)
    print(f"[{args.model.upper()}] FINAL TEST RESULTS")
    print(f"mAP50:        {mAP_dict['map_50'].item():.4f}")
    print(f"mean_IoU:     {float(np.mean(ious)):.4f}")
    print(f"Mean Latency: {float(np.mean(latencies)):.2f} ms")
    print("="*40)
    
    # Save Image Previews
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating Previews in {out_dir}...")
    
    with torch.no_grad():
        batch = next(iter(DataLoader(test_dataset, batch_size=4, shuffle=True)))
        x = batch["image"].to(device)
        box_preds, cls_logits = model(x)
        p_l, p_s, p_b = decode_batch_predictions(box_preds, cls_logits)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            img_np = x[i].cpu().permute(1, 2, 0).numpy()
            score = p_s[i]
            label = p_l[i]
            x1, y1, x2, y2 = p_b[i].cpu().tolist()
            w_b, h_b = max(0.0, x2 - x1), max(0.0, y2 - y1)
            gesture = TARGET_CLASSES[label]
            
            print(f"Preview {i+1}: y_i = ({score:.3f}, [{x1:.3f}, {y1:.3f}, {w_b:.3f}, {h_b:.3f}], {label + 1})  # {gesture}")
            
            ax = axes[i]
            ax.imshow(img_np)
            ax.add_patch(plt.Rectangle((x1*384, y1*384), w_b*384, h_b*384, fill=False, color='lime', linewidth=2))
            ax.set_title(f"{gesture} ({score:.2f})")
            ax.axis('off')
            
        plt.tight_layout()
        out_path = out_dir / f"{args.model}_preview.png"
        plt.savefig(out_path)
        print(f"Saved formal visual preview to {out_path}")

if __name__ == "__main__":
    main()
