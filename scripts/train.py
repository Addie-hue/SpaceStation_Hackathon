from ultralytics import YOLO
import torch

def main():
    print("🚀 SPACE STATION SAFETY DETECTION - MEMORY OPTIMIZED 🚀")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Start with smaller model to avoid memory issues
    model = YOLO("yolov8s.pt")
    
    # Memory-optimized parameters
    results = model.train(
        data="yolo_params.yaml",
        epochs=100,              # More epochs to compensate for smaller model
        imgsz=640,
        batch=8,                # Reduced for memory
        device=0,
        workers=0,              # CRITICAL: Disable multiprocessing
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        # Reduced augmentations for stability
        hsv_h=0.015,
        hsv_s=0.7, 
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        mosaic=0.5,            # Reduced mosaic
        mixup=0.05,            # Reduced mixup
        
        # Performance
        amp=True,
        cache=False,
        single_cls=False,
        overlap_mask=False,
        
        # Validation
        val=True,
        save=True,
        save_period=10,
    )
    
    print("🎯 TRAINING COMPLETE! READY FOR EVALUATION!")
    return results

if __name__ == "__main__":
    main()