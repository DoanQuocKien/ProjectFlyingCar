# Multi-Environment Setup Guide

## Overview
The `main.ipynb` notebook now supports execution on three platforms:
- **Local Machine** (Windows/Mac/Linux with CPU or GPU)
- **Google Colab** (GPU-enabled cloud notebooks)
- **Kaggle** (GPU-enabled notebooks with dataset access)

## Environment-Specific Changes

### 1. Environment Detection (Cell 2)

The notebook now automatically detects the execution environment using:

```python
def get_environment() -> str:
    # Checks for:
    # - Kaggle: /kaggle/input directory or KAGGLE_DATA_FOLDER env var
    # - Colab: google.colab module availability
    # - Local: Default fallback
    ...
```

**Detected environment: `ENVIRONMENT` variable**

### 2. Path Configuration (Cell 2)

Paths are automatically set based on the detected environment:

| Environment | Dataset Root | Model Directory | Output Directory |
|---|---|---|---|
| **Local** | `./data/hagrid-sample-30k-384p-5class` | `./models` | `.` (repo root) |
| **Colab** | `/content/hagrid_dataset/hagrid-sample-30k-384p` | `/content/models` | `/content` |
| **Kaggle** | `/kaggle/input/datasets/kinonquc/hagrid-dataset/hagrid-sample-30k-384p` | `/kaggle/working/models` | `/kaggle/working` |

### 3. Environment Validation (Cell 3 - NEW)

A comprehensive diagnostic cell validates:
- ✓ Environment detection
- ✓ Path configuration
- ✓ Hardware status (GPU/CPU)
- ✓ Dataset structure
- ✓ Required packages

Run this cell first to verify all systems are ready.

### 4. Dataset Loading (Cell 6)

**Updated to support:**
- Kaggle's full HaGrid dataset with 5-class filtering
- Local filtered dataset backup
- Automatic environment detection for data paths

**Key features:**
- Automatic filename indexing (works for Kaggle's large dataset)
- 5-class filtering: `one`, `peace`, `three`, `four`, `fist`
- Single-hand gesture filtering (images with exactly one hand)
- Detailed parsing diagnostics

### 5. DataLoader Configuration (Cell 8)

**Environment-optimized num_workers:**
- Windows: `num_workers=0` (multiprocessing issues)
- Kaggle: `num_workers=2` (safe for cloud environment)
- Colab: `num_workers=2` (safe for cloud environment)
- Linux/Mac: `num_workers=2` (if not on Kaggle/Colab)

### 6. Training Cell (Cell 17)

**Improvements:**
- Environment-aware device detection
- Proper GPU availability checking
- Environment-specific logging
- Model saved to environment-specific path

### 7. Model Download/Save Handler (Cell 19)

**Environment-specific handling:**
- **Colab**: Downloads model to local machine via `google.colab.files`
- **Kaggle**: Saves to `/kaggle/working` (accessible from Outputs tab)
- **Local**: Saves to `./models` (repository directory)

### 8. Inference Cell (Cell 23)

**Multi-environment support:**
- Loads model from environment-specific path
- Detects GPU availability
- Handles different output directories
- Saves visualizations to appropriate location
- Works on all three platforms

## Running the Notebook

### On Local Machine
```bash
# Simply run all cells in order
# Dataset should be in: ./data/hagrid-sample-30k-384p-5class
```

### On Google Colab
```
1. Open notebook in Colab
2. Upload/mount your dataset or use Kaggle API:
   !kaggle datasets download -d kinonquc/hagrid-dataset
   
3. Run all cells - paths will auto-configure
4. Model will be ready to download after training
```

### On Kaggle
```
1. Create a Kaggle notebook in a workspace with this notebook
2. Add dataset: "HaGrid Sample 30k 384p" by kinonquc
3. Run all cells - paths will auto-configure
4. Model saved to /kaggle/working/ (accessible from Outputs)
```

## Dataset Format Support

### Kaggle HaGrid Structure
```
/kaggle/input/datasets/kinonquc/hagrid-dataset/hagrid-sample-30k-384p/
├── ann_train_val/
│   ├── one.json
│   ├── peace.json
│   ├── three.json
│   ├── four.json
│   ├── fist.json
│   └── [other classes - ignored]
└── hagrid_30k/
    ├── train_val_one/
    ├── train_val_peace/
    ├── train_val_three/
    ├── train_val_four/
    └── train_val_fist/
```

### Local Dataset Structure (Same)
```
./data/hagrid-sample-30k-384p-5class/
├── ann_train_val/
│   ├── one.json
│   ├── peace.json
│   ├── three.json
│   ├── four.json
│   └── fist.json
└── hagrid_30k/
    ├── train_val_one/
    ├── train_val_peace/
    ├── train_val_three/
    ├── train_val_four/
    └── train_val_fist/
```

## Hardware Requirements

| Environment | Minimum | Recommended |
|---|---|---|
| **Local (CPU)** | 8GB RAM | 16GB RAM |
| **Local (GPU)** | 4GB VRAM | 8GB+ VRAM |
| **Colab** | Auto (TPU/GPU) | GPU (faster) |
| **Kaggle** | Auto (GPU) | GPU available by default |

## Troubleshooting

### Dataset not found on Kaggle
- Ensure the dataset "HaGrid Sample 30k 384p" is added to your Kaggle notebook
- Check `/kaggle/input/datasets/` exists
- Run the validation cell to see detailed diagnostic info

### Out of memory on Kaggle
- Reduce `BATCH_SIZE` from 16 to 8 or 4
- Reduce `NUM_WORKERS` from 2 to 0
- These are set in Cell 8 (DataLoader configuration)

### Slow training on local machine
- Training on CPU will be ~10x slower than GPU
- Model validation runs on entire validation set each epoch
- Consider reducing dataset size for testing

### Module not found errors
- All required packages should be auto-installed
- If errors persist, manually install:
  - `pip install torch torchvision` (local/Colab)
  - Packages pre-installed on Kaggle

### Path errors
- Run Cell 3 (Validation) to see actual paths being used
- Verify dataset exists at the shown path
- Check file permissions (especially on Kaggle)

## Key Variables

After running all setup cells, these variables are available:

```python
ENVIRONMENT          # 'kaggle', 'colab', or 'local'
DATASET_ROOT        # Path to HaGrid dataset
MODEL_DIR           # Path where models are saved
OUTPUT_DIR          # Path for output files
TARGET_CLASSES      # ['one', 'peace', 'three', 'four', 'fist']
CLASS_TO_ID         # {'one': 0, 'peace': 1, ...}
IMG_SIZE            # 448 (image resolution)
BATCH_SIZE          # 16 (training batch size)
GRID_SIZE           # 14 (detection grid size)
device              # torch.device('cuda' or 'cpu')
```

## Performance Notes

### Training Time (1 epoch)
- Local (CPU): ~30-60 minutes
- Local (GPU): ~1-2 minutes
- Colab (GPU): ~1-2 minutes
- Kaggle (GPU): ~1-2 minutes

### Inference Time (per image)
- CPU: ~1-2 seconds
- GPU: ~50-100 ms

## Model Checkpoint Format

Saved checkpoints include:
- `model_state_dict`: Model weights
- `best_accuracy`: Validation accuracy achieved
- `class_to_id`: Class mapping
- `img_size`: Training image resolution
- `grid_size`: Detection grid size
- `model_name`: Architecture identifier

## Future Improvements

- [ ] Add support for real-time webcam inference
- [ ] Support for continuous learning across notebooks
- [ ] Model ensemble for better accuracy
- [ ] Export to ONNX/TensorFlow formats
