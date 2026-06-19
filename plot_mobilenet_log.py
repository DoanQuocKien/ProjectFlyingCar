import matplotlib.pyplot as plt
import re
from pathlib import Path

log_file = "mobilenet_log.txt"

epochs = []
losses = []
maps = []

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        if "Epoch" in line and "Loss:" in line:
            try:
                loss_match = re.search(r"Loss:\s*([\d\.]+)", line)
                map_match = re.search(r"Val mAP50:\s*([\d\.]+)", line)
                
                if loss_match and map_match:
                    losses.append(float(loss_match.group(1)))
                    maps.append(float(map_match.group(1)))
                    epochs.append(len(epochs) + 1)
            except Exception as e:
                pass

out_dir = Path("models/mobilenet_ssd")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, color='tab:red', marker='o', linewidth=2)
plt.title('MobileNet SSD Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
loss_out_path = out_dir / "mobilenet_loss_plot.png"
plt.tight_layout()
plt.savefig(loss_out_path)
plt.close()

# 2. Plot mAP50
plt.figure(figsize=(10, 6))
plt.plot(epochs, maps, color='tab:blue', marker='s', linewidth=2)
plt.title('MobileNet SSD Validation mAP50', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('mAP50', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
map_out_path = out_dir / "mobilenet_map50_plot.png"
plt.tight_layout()
plt.savefig(map_out_path)
plt.close()

print(f"Saved separate plots to:")
print(f" - {loss_out_path}")
print(f" - {map_out_path}")
