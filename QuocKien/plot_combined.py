import matplotlib.pyplot as plt
import re
from pathlib import Path

def plot_combined_log(log_file, model_name, out_dir):
    epochs = []
    losses = []
    maps = []
    ious = []

    if not Path(log_file).exists():
        print(f"Skipping {log_file} (Not found)")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if "Epoch" in line and "Loss:" in line:
                try:
                    loss_match = re.search(r"Loss:\s*([\d\.]+)", line)
                    map_match = re.search(r"Val mAP50:\s*([\d\.]+)", line)
                    iou_match = re.search(r"Val mean_IoU:\s*([\d\.]+)", line)
                    
                    if loss_match and map_match and iou_match:
                        losses.append(float(loss_match.group(1)))
                        maps.append(float(map_match.group(1)))
                        ious.append(float(iou_match.group(1)))
                        epochs.append(len(epochs) + 1)
                except Exception as e:
                    pass

    if not epochs:
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', color=color, fontsize=12)
    ax1.plot(epochs, losses, color=color, marker='o', label='Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    color3 = 'tab:green'
    ax2.set_ylabel('Metrics (mAP50 & mean_IoU)', color='black', fontsize=12)  
    ax2.plot(epochs, maps, color=color2, marker='s', label='Val mAP50', linewidth=2)
    ax2.plot(epochs, ious, color=color3, marker='^', label='Val mean_IoU', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='black')

    fig.tight_layout()
    plt.title(f'{model_name} Training Progress (Loss, mAP50, mean_IoU)', fontsize=14)
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", bbox_to_anchor=(0.95, 0.5))

    save_path = out_path / f"{model_name.lower().replace(' ', '_')}_combined_plot.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined plot to {save_path}")

plot_combined_log("mobilenet_log.txt", "MobileNet_SSD", "models/mobilenet_ssd")
plot_combined_log("resnet_log.txt", "ResNet18", "models/resnet18")
