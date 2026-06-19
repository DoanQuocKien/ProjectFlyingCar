SEED = 24520789

# Reproducibility enforcement and quick seed check
import os, random, numpy as np, torch

# Ensure Python hash seed is stable for reproducible hashing behavior
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Prefer deterministic algorithms where possible (can reduce throughput)
try:
    torch.use_deterministic_algorithms(True)
    print('torch.use_deterministic_algorithms(True) enabled')
except Exception as e:
    print('Could not enable torch.use_deterministic_algorithms:', e)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Falling back to cudnn deterministic mode')

# Quick seed check: sample twice after reseeding and compare

def _sample_once():
    return {
        'py_random': [random.random() for _ in range(3)],
        'np_rand': np.random.rand(3).tolist(),
        'torch_cpu': torch.rand(3).tolist(),
    }

a = _sample_once()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
b = _sample_once()
print('Seed check match:', a == b)
print('Sample (1):', a)
print('Sample (2):', b)

def build_filename_index(img_root: Path, class_names=None):
    img_root = Path(img_root)
    index = {}
    if not img_root.exists():
        return index

    if class_names:
        candidate_dirs = [img_root / f"train_val_{class_name}" for class_name in class_names]
        candidate_dirs = [d for d in candidate_dirs if d.exists()]
    else:
        candidate_dirs = [img_root]

    for root_dir in candidate_dirs:
        for p in root_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                stem = p.stem
                name = p.name
                index.setdefault(stem, p)
                index.setdefault(name, p)
                try:
                    rel = str(p.relative_to(img_root))
                    index.setdefault(rel, p)
                except Exception:
                    pass
    return index


def resolve_image_path(filename_index: dict, image_key: str):
    if image_key is None:
        return None
    key = str(image_key)
    if key in filename_index:
        return filename_index[key]
    for ext in IMG_EXTS:
        if (key + ext) in filename_index:
            return filename_index[key + ext]
    stem = Path(key).stem
    if stem in filename_index:
        return filename_index[stem]
    return None


def xywh_to_xyxy(box, w: int, h: int, normalized: bool = True, xywh_mode: str = "topleft"):
    x, y, bw, bh = box[:4]
    if normalized:
        x = float(x) * w
        y = float(y) * h
        bw = float(bw) * w
        bh = float(bh) * h

    if xywh_mode == "center":
        x1 = float(x - bw / 2.0)
        y1 = float(y - bh / 2.0)
        x2 = float(x + bw / 2.0)
        y2 = float(y + bh / 2.0)
    else:
        # HaGRID bboxes are top-left xywh in this project dataset.
        x1 = float(x)
        y1 = float(y)
        x2 = float(x + bw)
        y2 = float(y + bh)

    x1 = max(0.0, min(x1, w))
    y1 = max(0.0, min(y1, h))
    x2 = max(0.0, min(x2, w))
    y2 = max(0.0, min(y2, h))
    return x1, y1, x2, y2

# Align downstream cells with the dataset prepared in cell 2.
EXTRACT_DIR = DATASET_ROOT
print("Using dataset root:", EXTRACT_DIR)
print("JSON files:", len(list(EXTRACT_DIR.rglob("*.json"))))
print("Image files:", len(list(EXTRACT_DIR.rglob("*.jpg"))))

def parse_hagrid_like_annotations(dataset_root: Path, target_classes):
    """Parse the exact HaGRID 5-class repo layout into flat training records.

    The dataset layout is:
    - ann_train_val/{class}.json
    - hagrid_30k/train_val_{class}/<image_id>.jpg

    Each JSON is a dict keyed by image id, and each value contains:
    - bboxes: list[[x, y, w, h]] (top-left xywh)
    - labels: list[str]
    - user_id: str (for better data splitting)
    """
    ann_dir = dataset_root / "ann_train_val"
    img_root = dataset_root / "hagrid_30k"
    json_files = [ann_dir / f"{class_name}.json" for class_name in target_classes]

    print(f"Found {len(json_files)} annotation files")
    print("Building filename index for target classes only...")
    filename_index = build_filename_index(img_root, target_classes)
    print(f"Indexed {len(filename_index)} image-name entries")

    records = []
    image_size_cache = {}
    t_start = time.perf_counter()
    t_last_report = t_start
    skipped_missing_image = 0
    skipped_bad_json = 0
    total_images_seen = 0

    for jf in tqdm(json_files, desc="Parsing annotations"):
        if not jf.exists():
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
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

            total_images_seen += 1
            img_path = resolve_image_path(filename_index, image_key)
            if img_path is None:
                skipped_missing_image += 1
                continue

            try:
                if img_path in image_size_cache:
                    w, h = image_size_cache[img_path]
                else:
                    # Read only image headers for size; avoid full decode during parsing.
                    with Image.open(img_path) as img_pil:
                        w, h = img_pil.size
                    image_size_cache[img_path] = (w, h)
            except Exception:
                continue

            # ADDED: Capture user_id for better data splitting
            user_id = item.get("user_id", "unknown")

            for label, box in zip(labels, bboxes):
                if label not in target_classes:
                    continue
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                normalized = max(box[:4]) <= 1.5
                x1, y1, x2, y2 = xywh_to_xyxy(box[:4], w, h, normalized=normalized, xywh_mode="topleft")
                records.append({
                    "image_path": str(img_path),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "class_id": CLASS_TO_ID[label],
                    "user_id": user_id,  # ADDED: user_id for splitting
                })

            if len(records) % 5000 == 0 and len(records) > 0:
                now = time.perf_counter()
                elapsed = now - t_start
                since_last = now - t_last_report
                t_last_report = now
                print(f"  Parsed {len(records):,} records from {total_images_seen:,} images in {elapsed:.1f}s (+{since_last:.1f}s)")

    total_elapsed = time.perf_counter() - t_start
    print(f"Skipped missing-image records: {skipped_missing_image}")
    print(f"Skipped unreadable JSON files: {skipped_bad_json}")
    print(f"  Unique images cached: {len(image_size_cache)}")
    print(f"  Total image keys visited: {total_images_seen}")
    print(f"  Total parse time: {total_elapsed:.1f}s")
    return records


records = parse_hagrid_like_annotations(EXTRACT_DIR, TARGET_CLASSES)
print("Parsed records (raw):", len(records))
# Filter to single-hand images only (images that appear exactly once)
img_counts = Counter(r['image_path'] for r in records)
single_records = [r for r in records if img_counts[r['image_path']] == 1]
print(f"Records before filter: {len(records)}, after filter (single-hand only): {len(single_records)}")
records = single_records
if len(records) == 0:
    print("No samples parsed after single-hand filtering. Check dataset structure and annotation schema.")

# Use the exact filtered 5-class dataset already prepared in the repo.
# Split by user_id to avoid data leakage (no user should appear in multiple splits)
print("Checking if records have user_id field...")
has_user_id = any('user_id' in r for r in records)

if has_user_id:
    # User-based splitting
    unique_users = list(set([r.get('user_id', 'unknown') for r in records]))
    random.shuffle(unique_users)
    n_users = len(unique_users)
    n_train_users = int(0.7 * n_users)
    n_val_users = int(0.15 * n_users)
    
    train_users = set(unique_users[:n_train_users])
    val_users = set(unique_users[n_train_users:n_train_users + n_val_users])
    test_users = set(unique_users[n_train_users + n_val_users:])
    
    train_records = [r for r in records if r.get('user_id', 'unknown') in train_users]
    val_records = [r for r in records if r.get('user_id', 'unknown') in val_users]
    test_records = [r for r in records if r.get('user_id', 'unknown') in test_users]
    
    print(f"Split by user ID: {len(train_users)} train users, {len(val_users)} val users, {len(test_users)} test users")
else:
    # Fallback to random shuffling if user_id not available (legacy dataset)
    print("WARNING: user_id not available in records. Using random splitting (potential data leakage).")
    random.shuffle(records)
    
    n = len(records)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

print(f"Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")


def draw_box(image, box, color, width=2):
    draw = ImageDraw.Draw(image)
    for offset in range(width):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=color,
        )


def show_samples(samples, n_show=6):
    if len(samples) == 0:
        print("No samples to show")
        return
    idxs = np.random.choice(len(samples), size=min(n_show, len(samples)), replace=False)
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(idxs, 1):
        s = samples[idx]
        with Image.open(s["image_path"]) as img_pil:
            img = img_pil.convert("RGB")
        x1, y1, x2, y2 = map(int, s["bbox_xyxy"])
        draw_box(img, (x1, y1, x2, y2), color=(0, 255, 0), width=2)
        plt.subplot(2, 3, i)
        plt.imshow(img)
        plt.title(ID_TO_CLASS[s["class_id"]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_samples(train_records, n_show=6)

from torchvision import models
import torch.nn as nn
import torch

class ResNetLocalization(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, 4 + num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.pool(feat).flatten(1)
        out = self.head(feat)
        box_preds = torch.sigmoid(out[:, :4])
        cls_logits = out[:, 4:]
        return box_preds, cls_logits

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

clean_model = ResNetLocalization(num_classes=len(TARGET_CLASSES)).to(device)


@torch.no_grad()
def decode_single_prediction(obj_logits, box_preds, cls_logits, w, h, grid_size: int | None = None):
    """
    Decodes the single most confident prediction from the model's grid output.
    Returns: [x1, y1, x2, y2], class_id, confidence_score
    """
    if grid_size is None:
        grid_size = obj_logits.shape[-1]

    obj_probs = torch.sigmoid(obj_logits[0])
    flat_idx = torch.argmax(obj_probs)
    py, px = int(flat_idx // grid_size), int(flat_idx % grid_size)
    score = obj_probs[py, px].item()
    rel = box_preds[0, py, px]
    cx = (px + rel[0]) / grid_size
    cy = (py + rel[1]) / grid_size
    bw, bh = rel[2], rel[3]
    x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
    x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
    class_id = torch.argmax(cls_logits[0, py, px]).item()
    return [float(x1), float(y1), float(x2), float(y2)], int(class_id), float(score)

from torchvision.ops import nms

@torch.no_grad()
def decode_and_nms(obj_logits, box_preds, cls_logits, grid_size: int | None = None, score_threshold=0.5, iou_threshold=0.4):
    """
    Decodes all grid cells and applies Non-Maximum Suppression.
    This is essential for live detection where a hand might overlap multiple grid cells.
    """
    if grid_size is None:
        grid_size = obj_logits.shape[-1]

    batch_size = obj_logits.size(0)
    final_preds = []
    obj_probs = torch.sigmoid(obj_logits)

    for b in range(batch_size):
        probs = obj_probs[b].view(-1)
        mask = probs > score_threshold

        if not mask.any():
            final_preds.append(None)
            continue

        filtered_probs = probs[mask]
        indices = torch.nonzero(mask).squeeze(1)
        py = indices // grid_size
        px = indices % grid_size
        rel_boxes = box_preds[b, py, px]
        p_cx = (px.float() + rel_boxes[:, 0]) / grid_size
        p_cy = (py.float() + rel_boxes[:, 1]) / grid_size
        p_w, p_h = rel_boxes[:, 2], rel_boxes[:, 3]
        boxes = torch.stack([
            p_cx - p_w / 2, p_cy - p_h / 2,
            p_cx + p_w / 2, p_cy + p_h / 2
        ], dim=1).clamp(0, 1)
        keep_idx = nms(boxes, filtered_probs, iou_threshold)
        class_ids = torch.argmax(cls_logits[b, py, px], dim=1)
        final_preds.append({
            "boxes": boxes[keep_idx],
            "scores": filtered_probs[keep_idx],
            "labels": class_ids[keep_idx]
        })

    return final_preds

print("NMS post-processing logic integrated for better live detection stability.")

from google.colab import files

# Path to the trained model file
model_to_download = MODEL_DIR / "resnet18_hagrid_detector.pt"

# Check if the file exists before attempting to download
if model_to_download.exists():
    files.download(str(model_to_download))
    print(f"'{model_to_download.name}' downloaded successfully to your local machine.")
else:
    print(f"Model file not found at {model_to_download}. Please ensure the model training and saving steps were completed successfully.")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

@torch.no_grad()
def final_report(model, loader):
    model.eval()
    gt_labels, pred_labels = [], []
    for batch in tqdm(loader):
        x = batch["image"].to(device)
        box_preds, cls_logits = model(x)
        p_l, p_s, p_b = decode_batch_predictions(box_preds, cls_logits)
        g_l = batch["target_label"].cpu().tolist()
        gt_labels.extend(g_l)
        pred_labels.extend(p_l)

    print("\n--- Classification Report ---")
    print(classification_report(gt_labels, pred_labels, target_names=TARGET_CLASSES))

    cm = confusion_matrix(gt_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.show()

if 'clean_model' in globals():
    final_report(clean_model, test_loader)
else:
    print("Model not trained yet.")
# Save split indices, environment details, and evaluation artifacts
import json, platform, subprocess, sys
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

REPORT_DIR = MODEL_DIR
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Persist train/val/test splits
splits = {
    'train': [r['image_path'] for r in train_records],
    'val': [r['image_path'] for r in val_records],
    'test': [r['image_path'] for r in test_records],
}
with open(REPORT_DIR / 'resnet18_splits.json', 'w', encoding='utf-8') as f:
    json.dump(splits, f, indent=2)
print(f"Saved splits to: {REPORT_DIR / 'resnet18_splits.json'}")

# 2) Record environment details and pinned package snapshot
env_info = {
    'python': sys.version,
    'platform': platform.platform(),
    'torch': torch.__version__,
}
try:
    pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.DEVNULL).decode('utf-8')
    env_info['pip_freeze'] = pip_freeze
except Exception as e:
    env_info['pip_freeze'] = f'pip freeze failed: {e}'

with open(REPORT_DIR / 'resnet18_env_info.txt', 'w', encoding='utf-8') as f:
    for k, v in env_info.items():
        if k == 'pip_freeze':
            f.write('\n== pip freeze ==\n')
            f.write(v)
        else:
            f.write(f"{k}: {v}\n")
print(f"Saved environment info to: {REPORT_DIR / 'resnet18_env_info.txt'}")

# 3) Save classification report + confusion matrix from test predictions
if 'clean_model' in globals() and 'test_loader' in globals():
    clean_model.eval()
    all_preds, all_gts = [], []
    for batch in tqdm(test_loader, leave=False):
        x = batch['image'].to(device)
        box_preds, cls_logits = clean_model(x)
        p_l, p_s, p_b = decode_batch_predictions(box_preds, cls_logits)
        all_preds.extend(p_l)
        all_gts.extend(batch["target_label"].cpu().tolist())

    clf_report = classification_report(all_gts, all_preds, target_names=TARGET_CLASSES, output_dict=True, zero_division=0)
    with open(REPORT_DIR / 'resnet18_classification_report.json', 'w', encoding='utf-8') as f:
        json.dump(clf_report, f, indent=2)
    print(f"Saved classification report to: {REPORT_DIR / 'resnet18_classification_report.json'}")

    cm = confusion_matrix(all_gts, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('ResNet-18 Confusion Matrix')
    cm_path = REPORT_DIR / 'resnet18_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved confusion matrix image to: {cm_path}")

    if 'history_df' in globals():
        history_csv = REPORT_DIR / 'resnet18_training_history.csv'
        history_df.to_csv(history_csv, index=False)
        print(f"Saved training history to: {history_csv}")

    summary = {
        'best_model_path': str(MODEL_PATH) if 'MODEL_PATH' in globals() else str(REPORT_DIR / 'resnet18_hagrid_detector.pt'),
        'model_exists': (MODEL_PATH.exists() if 'MODEL_PATH' in globals() else (REPORT_DIR / 'resnet18_hagrid_detector.pt').exists()),
        'splits_file': str(REPORT_DIR / 'resnet18_splits.json'),
        'env_info_file': str(REPORT_DIR / 'resnet18_env_info.txt'),
        'classification_report': str(REPORT_DIR / 'resnet18_classification_report.json'),
        'confusion_matrix': str(REPORT_DIR / 'resnet18_confusion_matrix.png'),
    }
    with open(REPORT_DIR / 'resnet18_report_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved report summary to: {REPORT_DIR / 'resnet18_report_summary.json'}")
else:
    print('Model or test_loader not available in the notebook state; run training and evaluation cells first.')

@torch.no_grad()
def visualize_predictions(model, dataset, n_show=8, score_thr=0.1, iou_thr=0.4):
    if len(dataset) == 0:
        print("Dataset empty")
        return

    model.eval()
    idxs = np.random.choice(len(dataset), size=min(n_show, len(dataset)), replace=False)
    plt.figure(figsize=(20, 12))

    for i, idx in enumerate(idxs, 1):
        sample = dataset.records[idx]
        with Image.open(sample["image_path"]) as img_pil:
            img = img_pil.convert("RGB")
            w, h = img.size

            # Prep input
            resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            x = np.asarray(resized, dtype=np.float32) / 255.0
            xt = torch.tensor(np.transpose(x, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)

            # Inference
            box_preds, cls_logits = model(xt)
            p_l, p_s, p_b = decode_batch_predictions(box_preds, cls_logits)

            # Draw Ground Truth (Green)
            gx1, gy1, gx2, gy2 = map(int, sample["bbox_xyxy"])
            draw_box(img, (gx1, gy1, gx2, gy2), color=(0, 255, 0), width=3)

            pred_text = "No Detection"
            if len(p_b) > 0:
                box = p_b[0].cpu().numpy()
                score = p_s[0]
                label = p_l[0]

                # RESCALE TO PIXELS
                px1, py1 = int(box[0] * w), int(box[1] * h)
                px2, py2 = int(box[2] * w), int(box[3] * h)

                draw_box(img, (px1, py1, px2, py2), color=(255, 0, 0), width=5)
                pred_text = f"Pred: {ID_TO_CLASS[label]} ({score:.2f})"

            plt.subplot((n_show + 3) // 4, 4, i)
            plt.imshow(img)
            plt.title(f"GT: {ID_TO_CLASS[sample['class_id']]}\n{pred_text}")
            plt.axis("off")

    plt.tight_layout()
    plt.show()

# Re-running visualization with red prediction boxes
visualize_predictions(clean_model, test_ds, n_show=8)
# Re-running visualization with red prediction boxes and score threshold 0.3
visualize_predictions(clean_model, test_ds, n_show=8, score_thr=0.3)

# Evaluate the model on the test set
print("Evaluating ResNet model on test data...")
test_metrics = evaluate_model(clean_model, test_loader)

# Display metrics nicely
metrics_df = pd.DataFrame([test_metrics])
display(metrics_df)

def plot_history(df):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df.get('mAP50', df.get('accuracy')), label='Val mAP50')
    plt.plot(df['epoch'], df['mean_iou'], label='Val IoU')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

if 'history_df' in globals():
    plot_history(history_df)
else:
    print("History not found. Ensure the training cell was executed.")
