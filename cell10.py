IMG_SIZE = 320
BATCH_SIZE = 32
GRID_SIZE = 10  # 320 / 32 downsampling ratio

# Optimize num_workers based on environment and OS
# Windows: always use 0 (multiprocessing issues)
# Linux/Mac on Kaggle/Colab: use 2 for cloud throughput
if ENVIRONMENT in ["kaggle", "colab"]:
    NUM_WORKERS = 2
else:
    import platform
    if platform.system() == "Windows":
        NUM_WORKERS = 0
    else:
        NUM_WORKERS = 2

print(f"DataLoader configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"  Grid size: {GRID_SIZE}")
print(f"  Num workers: {NUM_WORKERS}")
print(f"  Environment: {ENVIRONMENT}")

def normalize_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    return [x1 / w, y1 / h, x2 / w, y2 / h]

class GestureGridDataset(Dataset):
    def __init__(self, records, img_size=320, grid_size=10, augment=False):
        self.records = records
        self.img_size = img_size
        self.grid_size = grid_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        with Image.open(r["image_path"]) as img_pil:
            img = img_pil.convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = r["bbox_xyxy"]

            if self.augment:
                if random.random() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    new_x1 = w - x2
                    new_x2 = w - x1
                    x1, x2 = new_x1, new_x2
                if random.random() > 0.5:
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.8, 1.2))

            x1_n, y1_n, x2_n, y2_n = normalize_box_xyxy([x1, y1, x2, y2], w, h)
            img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))

            target = torch.zeros((self.grid_size, self.grid_size, 6), dtype=torch.float32)
            cx, cy = (x1_n + x2_n) / 2.0, (y1_n + y2_n) / 2.0
            box_w, box_h = max(0.0, x2_n - x1_n), max(0.0, y2_n - y1_n)

            cell_x = min(max(int(cx * self.grid_size), 0), self.grid_size - 1)
            cell_y = min(max(int(cy * self.grid_size), 0), self.grid_size - 1)

            rel_cx = (cx * self.grid_size) - cell_x
            rel_cy = (cy * self.grid_size) - cell_y

            target[cell_y, cell_x, 0] = 1.0
            target[cell_y, cell_x, 1:5] = torch.tensor([rel_cx, rel_cy, box_w, box_h], dtype=torch.float32)
            target[cell_y, cell_x, 5] = float(r["class_id"])

        return {"image": torch.tensor(img, dtype=torch.float32), "target": target}

train_ds = GestureGridDataset(train_records, img_size=IMG_SIZE, grid_size=GRID_SIZE, augment=True)
val_ds = GestureGridDataset(val_records, img_size=IMG_SIZE, grid_size=GRID_SIZE, augment=False)
test_ds = GestureGridDataset(test_records, img_size=IMG_SIZE, grid_size=GRID_SIZE, augment=False)

# Create DataLoaders with environment-optimized settings
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"\n✓ DataLoaders created:")
print(f"  Train batches: {len(train_loader)} ({len(train_ds)} samples)")
print(f"  Val batches: {len(val_loader)} ({len(val_ds)} samples)")
print(f"  Test batches: {len(test_loader)} ({len(test_ds)} samples)")