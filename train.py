import os
from ultralytics import YOLO

# --- Configurations ---
DS_PATH = 'data.yaml'
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 416
DEVICE = 'cuda'  
TRAINING_RESULTS_PATH = "TrainingResults"
MODEL_ARCH = "yolov5xu.pt"

def train_model():
    """Initializes, trains, and exports the YOLO face mask detection model."""
    print(f"[INFO] Initializing YOLO model ({MODEL_ARCH})...")
    model = YOLO(MODEL_ARCH)
    
    print("[INFO] Starting training...")
    model.train(
        data=DS_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=TRAINING_RESULTS_PATH,
        name="MaskDetection",
        exist_ok=True
    )
    
    print("[INFO] Training completed. Exporting best model to ONNX...")
    export_path = model.export(format="onnx", imgsz=IMG_SIZE)
    print(f"[INFO] Model successfully exported to: {export_path}")

if __name__ == "__main__":
    train_model()