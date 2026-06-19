SEED = 24520789

# Reproducibility enforcement and quick seed check
import os, random, numpy as np, torch

# Ensure PYTHON hash seed is stable
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Prefer deterministic algorithms when available, but do not fail on ops that lack
# deterministic CUDA implementations. The smoke-test cell below only needs to verify
# that the RNG stream is repeatable.
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
    print('torch.use_deterministic_algorithms(True, warn_only=True) enabled')
except TypeError:
    try:
        torch.use_deterministic_algorithms(True)
        print('torch.use_deterministic_algorithms(True) enabled')
    except Exception as e:
        print('Could not enable torch.use_deterministic_algorithms:', e)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('Falling back to cudnn deterministic mode')

# Quick seed check: sample twice after reseeding
def _sample_once():
    return {
        'py_random': [random.random() for _ in range(3)],
        'np_rand': np.random.rand(3).tolist(),
        'torch_cpu': torch.rand(3).tolist(),
    }

a = _sample_once()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
b = _sample_once()
print('Seed check match:', a == b)
print('Sample (1):', a)
print('Sample (2):', b)

KEEP_CLASSES = ["one", "peace", "three", "four", "fist"]
TARGET_CLASSES = KEEP_CLASSES
CLASS_TO_ID = {c: i for i, c in enumerate(TARGET_CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print(f"Target classes: {TARGET_CLASSES}")
print(f"Class to ID mapping: {CLASS_TO_ID}")

def find_json_files(root: Path):
    return [p for p in root.rglob("*.json") if p.is_file()]

def build_filename_index(root: Path):
    """Scan the repo dataset once and index image stems to absolute paths."""
    filename_index = {}
    for img_path in root.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
            filename_index.setdefault(img_path.name, img_path)
            filename_index.setdefault(img_path.stem, img_path)
    return filename_index

def resolve_image_path(filename_index: dict, image_key: str):
    key = Path(image_key).stem
    return filename_index.get(key) or filename_index.get(Path(image_key).name)

def xywh_to_xyxy(box, w, h, normalized=True):
    x, y, bw, bh = box
    if normalized:
        x *= w
        y *= h
        bw *= w
        bh *= h
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(w - 1), x + bw)
    y2 = min(float(h - 1), y + bh)
    return [x1, y1, x2, y2]

def parse_hagrid_annotations(dataset_root: Path, target_classes, environment: str):
    """
    Parse HaGRID annotations based on environment.
    
    Local format:
    - ann_train_val/{class}.json
    - hagrid_30k/train_val_{class}/<image_id>.jpg
    
    Kaggle format (full dataset):
    - ann_train_val/{class}.json  (same structure as local)
    - hagrid_30k/train_val_{class}/<image_id>.jpg (same structure as local)
    
    Returns list of records with fields: image_path, bbox_xyxy, class_id
    """
    ann_dir = dataset_root / "ann_train_val"
    img_root = dataset_root / "hagrid_30k"
    
    print(f"Looking for annotations in: {ann_dir}")
    print(f"Looking for images in: {img_root}")
    
    # Find JSON files for target classes
    json_files = []
    for class_name in target_classes:
        json_path = ann_dir / f"{class_name}.json"
        if json_path.exists():
            json_files.append(json_path)
        else:
            print(f"Warning: {json_path} not found")
    
    print(f"Found {len(json_files)} annotation files for target classes")
    
    if not json_files:
        print("ERROR: No annotation files found! Check dataset structure.")
        return []
    
    print("Building filename index once (this may take a minute for Kaggle)...")
    filename_index = build_filename_index(img_root)
    print(f"Indexed {len(filename_index)} image-name entries")

    records = []
    skipped_missing_image = 0
    skipped_bad_json = 0
    skipped_non_target_class = 0

    for jf in tqdm(json_files, desc="Parsing annotations"):
        if not jf.exists():
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error reading {jf}: {e}")
            skipped_bad_json += 1
            continue

        if not isinstance(data, dict):
            skipped_bad_json += 1
            continue

        for image_key, item in data.items():
            if not isinstance(item, dict):
                continue

            labels = item.get("labels") or []
            bboxes = item.get("bboxes") or []
            if not labels or not bboxes:
                continue

            img_path = resolve_image_path(filename_index, image_key)
            if img_path is None:
                skipped_missing_image += 1
                continue

            try:
                with Image.open(img_path) as img_pil:
                    img_pil = img_pil.convert("RGB")
                    w, h = img_pil.size
            except Exception:
                continue

            for label, box in zip(labels, bboxes):
                if label not in target_classes:
                    skipped_non_target_class += 1
                    continue
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                normalized = max(box[:4]) <= 1.5
                x1, y1, x2, y2 = xywh_to_xyxy(box[:4], w, h, normalized=normalized)
                records.append({
                    "image_path": str(img_path),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "class_id": CLASS_TO_ID[label],
                })

    print(f"Parsing complete:")
    print(f"  Valid records: {len(records)}")
    print(f"  Skipped missing-image: {skipped_missing_image}")
    print(f"  Skipped unreadable JSON: {skipped_bad_json}")
    print(f"  Skipped non-target classes: {skipped_non_target_class}")
    
    return records

# Parse dataset with environment-aware handler
print("\n" + "="*60)
print(f"Parsing HaGRID dataset ({ENVIRONMENT} environment)")
print("="*60)

records = parse_hagrid_annotations(DATASET_ROOT, TARGET_CLASSES, ENVIRONMENT)
print(f"\nParsed records (raw): {len(records)}")

# Filter to single-hand images only (images that appear exactly once)
img_counts = Counter(r['image_path'] for r in records)
single_records = [r for r in records if img_counts[r['image_path']] == 1]
print(f"Records before filter: {len(records)}, after filter (single-hand only): {len(single_records)}")
records = single_records

if len(records) == 0:
    print("ERROR: No samples parsed after single-hand filtering. Check dataset structure and annotation schema.")
else:
    print(f"✓ Successfully parsed {len(records)} single-hand gesture samples")

from torchvision import models
import torch.nn as nn
import torch

class MobileNetLocalization(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights='DEFAULT')
        self.backbone = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # 4 box coordinates + num_classes
        self.head = nn.Linear(1280, 4 + num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        out = self.head(feat)
        box_preds = torch.sigmoid(out[:, :4]) # clamp to [0, 1]
        cls_logits = out[:, 4:]
        return box_preds, cls_logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

clean_model = MobileNetLocalization(num_classes=len(TARGET_CLASSES)).to(device)




# Debug cell: single-batch forward timing and basic checks
print("--- Debug: single-batch smoke test ---")
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())

try:
    batch = next(iter(train_loader))
except Exception as e:
    print("Failed to fetch a batch from train_loader:", e)
    raise

x = batch["image"].to(device)
target_boxes = batch["target_box"].to(device)
target_labels = batch["target_label"].to(device)
print("Batch shapes -> image:", x.shape, " target_boxes:", target_boxes.shape)

import time
if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
box_preds, cls_logits = clean_model(x)
if device.type == "cuda":
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"Forward time: {t1 - t0:.4f}s (batch size {x.size(0)})")

loss, l_box, l_cls = localization_loss(box_preds, cls_logits, target_boxes, target_labels)
print("Loss summary -> total:", float(loss), "box:", l_box, "cls:", l_cls)

print("box_preds stats -> mean, std:", float(box_preds.mean()), float(box_preds.std()))
print("cls_logits stats -> mean, std:", float(cls_logits.mean()), float(cls_logits.std()))
print("--- End debug ---")