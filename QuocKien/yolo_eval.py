## Tips for Using YOLOv11n Variant

### Why YOLOv11n (Nano)?
- **Speed**: Fastest YOLO model for real-time inference
- **Size**: Smallest weights (~2.6 MB), ideal for edge deployment
- **mAP50**: Surprisingly good for a nano model with proper training

### Why 80 Epochs?
- Default 50 epochs may be insufficient for nano to converge
- Extended training allows better weight optimization
- Early stopping (patience=20) prevents overfitting

### Best-Model Selection
- Ultralytics automatically saves best.pt during training
- Best model chosen based on validation fitness (mAP)
- Copied to MODEL_SAVE_PATH for easy access

### Compare with main_yolo26.ipynb
- main_yolo26: Auto-detection (tries YOLOv11n → YOLOv8n) with 50 epochs
- main_yolo11n: YOLOv11n hardcoded with 80 epochs
- This variant is optimized for speed; use if nano is your target

### Production Deployment
1. Export to ONNX for CPU inference
2. Use best.pt for edge devices (tiny size)
3. Benchmark latency on target hardware
4. Consider INT8 quantization for further speedup