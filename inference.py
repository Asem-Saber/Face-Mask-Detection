import argparse
from ultralytics import YOLO

def run_inference(model_path, source, conf_threshold, output_dir):
    """
    Runs YOLO inference on a specified source (image file, video file, or webcam id).
    """
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(model_path)
    
    if source.isdigit():
        source = int(source)
        is_webcam = True
    else:
        is_webcam = False

    print(f"[INFO] Running inference on source: {source}")
    
    model.predict(
        source=source,
        conf=conf_threshold,
        project=output_dir,
        name="live_predictions" if is_webcam else "test_predictions",
        save=True,          
        save_txt=True,      
        save_conf=True,    
        show=is_webcam      
    )
    
    print(f"[INFO] Inference completed. Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection Inference CLI")
    parser.add_argument("--model", type=str, default="TrainingResults/MaskDetection/weights/best.pt", 
                        help="Path to the trained YOLO weights (e.g., best.pt or best.onnx)")
    parser.add_argument("--source", type=str, default="0", 
                        help="Input source: '0' for webcam, or path to image/video folder/file")
    parser.add_argument("--conf", type=float, default=0.5, 
                        help="Confidence threshold for bounding boxes")
    parser.add_argument("--output", type=str, default="InferenceResults", 
                        help="Directory to save the prediction results")
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model, 
        source=args.source, 
        conf_threshold=args.conf, 
        output_dir=args.output
    )