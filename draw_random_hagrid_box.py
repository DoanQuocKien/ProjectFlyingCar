"""Draw one random HaGRID ground-truth box with a demo confidence label.

Example:
    python draw_random_hagrid_box.py
    python draw_random_hagrid_box.py --class-name peace --confidence 0.92
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASSES = ("one", "peace", "three", "four", "fist")
DEFAULT_DATASET = Path("data/hagrid-sample-30k-384p-5class")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--class-name", choices=CLASSES, help="Optional class to sample.")
    parser.add_argument("--confidence", type=float, default=0.92)
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible sampling.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_reports/random_hagrid_detection.jpg"),
    )
    return parser.parse_args()


def load_candidates(dataset: Path, class_name: str) -> list[tuple[str, list[float]]]:
    annotation_path = dataset / "ann_train_val" / f"{class_name}.json"
    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    candidates: list[tuple[str, list[float]]] = []
    for image_id, item in annotations.items():
        if not image_exists(dataset, class_name, image_id):
            continue
        for box, label in zip(item.get("bboxes", []), item.get("labels", [])):
            if label == class_name:
                candidates.append((image_id, box))
    return candidates


def image_exists(dataset: Path, class_name: str, image_id: str) -> bool:
    image_dir = dataset / "hagrid_30k" / f"train_val_{class_name}"
    return any((image_dir / f"{image_id}{suffix}").exists() for suffix in (".jpg", ".jpeg", ".png"))


def find_image(dataset: Path, class_name: str, image_id: str) -> Path:
    image_dir = dataset / "hagrid_30k" / f"train_val_{class_name}"
    for suffix in (".jpg", ".jpeg", ".png"):
        image_path = image_dir / f"{image_id}{suffix}"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"No image file found for {image_id} in {image_dir}")


def normalized_xywh_to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, min(width - 1, round(x * width)))
    y1 = max(0, min(height - 1, round(y * height)))
    x2 = max(0, min(width - 1, round((x + w) * width)))
    y2 = max(0, min(height - 1, round((y + h) * height)))
    return x1, y1, x2, y2


def draw_detection(
    image_path: Path,
    box: list[float],
    class_name: str,
    confidence: float,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x1, y1, x2, y2 = normalized_xywh_to_pixels(box, width, height)

    line_width = max(6, min(width, height) // 70)
    color = (0, 255, 90)
    label = f"{class_name} {confidence:.2f}"

    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    font = ImageFont.load_default(size=max(18, min(width, height) // 18))
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    pad = max(6, line_width)
    label_y1 = max(0, y1 - text_height - pad * 2)
    label_y2 = label_y1 + text_height + pad * 2

    draw.rectangle((x1, label_y1, x1 + text_width + pad * 2, label_y2), fill=color)
    draw.text((x1 + pad, label_y1 + pad), label, fill=(0, 0, 0), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    class_name = args.class_name or rng.choice(CLASSES)

    candidates = load_candidates(args.dataset, class_name)
    if not candidates:
        raise RuntimeError(f"No usable annotations found for class '{class_name}'.")

    image_id, box = rng.choice(candidates)
    image_path = find_image(args.dataset, class_name, image_id)
    draw_detection(image_path, box, class_name, args.confidence, args.output)

    print(f"image: {image_path}")
    print(f"class: {class_name}")
    print(f"confidence: {args.confidence:.2f}")
    print(f"bbox_xywh_normalized: {[round(value, 6) for value in box]}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
