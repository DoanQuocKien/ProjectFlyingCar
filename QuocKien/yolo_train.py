# Fine-tune YOLOv11n on the 5-class dataset
print(f"\n{'='*70}")
print("FINE-TUNING YOLOv11n ON HaGRID 5-CLASS DATASET")
print(f"{'='*70}")

# INCREASED EPOCHS FOR BETTER NANO CONVERGENCE
TRAIN_EPOCHS = 80
TRAIN_IMG_SIZE = 640
TRAIN_BATCH_SIZE = 16
TRAIN_LR = 1e-4
TRAIN_PATIENCE = 20  # Increased patience for longer training

MODEL_SAVE_PATH = MODEL_DIR / 'yolo11n_hagrid_best.pt'

print(f"Training Configuration:")
print(f"  Epochs: {TRAIN_EPOCHS} (INCREASED from 50 for nano convergence)")
print(f"  Image size: {TRAIN_IMG_SIZE}")
print(f"  Batch size: {TRAIN_BATCH_SIZE}")
print(f"  Learning rate: {TRAIN_LR}")
print(f"  Patience: {TRAIN_PATIENCE}")
print(f"  Best model tracking: ENABLED (validation accuracy)")

# Start training
print(f"\nStarting YOLOv11n training on {ENVIRONMENT}...")

results = yolo_model.train(
    data=str(yaml_path),
    epochs=TRAIN_EPOCHS,
    imgsz=TRAIN_IMG_SIZE,
    batch=TRAIN_BATCH_SIZE,
    lr0=TRAIN_LR,
    patience=TRAIN_PATIENCE,
    device=0 if torch.cuda.is_available() else 'cpu',
    project=str(MODEL_DIR / 'yolo_runs'),
    name='hagrid_yolo11n',
    exist_ok=True,
    verbose=True,
    save=True,
    val=True,
    plots=True,
    mosaic=1.0,
    flipud=0.5,
    fliplr=0.5,
)

print(f"\n{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
fitness_value = None
if hasattr(results, 'box') and hasattr(results.box, 'fitness'):
    try:
        fitness_value = results.box.fitness()
    except TypeError:
        fitness_value = results.box.fitness
elif hasattr(results, 'results_dict') and isinstance(getattr(results, 'results_dict'), dict):
    fitness_value = results.results_dict.get('fitness')

if fitness_value is not None:
    print(f'Best fitness achieved: {float(fitness_value):.4f}')
else:
    print('Best fitness achieved: unavailable in this Ultralytics version')

# COPY BEST MODEL (selected by Ultralytics based on validation accuracy)
trainer = getattr(yolo_model, 'trainer', None)
save_dir = Path(trainer.save_dir) if trainer is not None and getattr(trainer, 'save_dir', None) else (MODEL_DIR / 'yolo_runs' / 'hagrid_yolo11n')
best_model_path = save_dir / 'weights' / 'best.pt'
if best_model_path.exists():
    import shutil
    shutil.copy2(best_model_path, MODEL_SAVE_PATH)
    print(f'✓ Best model saved to: {MODEL_SAVE_PATH}')
    print(f'  (Model selected based on best validation accuracy across {TRAIN_EPOCHS} epochs)')
else:
    print('⚠ Note: Check training logs for best model path')
