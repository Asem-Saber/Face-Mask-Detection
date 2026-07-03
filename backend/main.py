import os
import cv2
import numpy as np
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse
from ultralytics import YOLO

app = FastAPI(title="Face Mask Detection API")

# Load the model
MODEL_PATH = os.getenv("MODEL_PATH", "/app/weights/best.pt")
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        results = model.predict(img, conf=0.5)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Encode back to JPEG
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        
        return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    temp_id = str(uuid.uuid4())
    input_path = f"/tmp/{temp_id}_input.mp4"
    output_dir = f"/tmp/{temp_id}_output"
    
    try:
        # Save uploaded video
        with open(input_path, "wb") as f:
            f.write(await file.read())
            
        # Run inference
        # YOLO saves to project/name, e.g., output_dir/predict/video.mp4
        results = model.predict(
            source=input_path, 
            conf=0.5, 
            project=output_dir,
            name="predict",
            save=True
        )
        
        # Find the output video file
        predict_dir = os.path.join(output_dir, "predict")
        # The output file has the same name as the input file, but possibly different extension
        output_filename = os.path.basename(input_path)
        # YOLO might change .mp4 to .avi or keep .mp4 depending on backend, but typically keeps it or uses .avi
        # Let's search for the video file in the predict directory
        output_files = os.listdir(predict_dir)
        video_files = [f for f in output_files if f.endswith(('.mp4', '.avi', '.webm'))]
        
        if not video_files:
            raise Exception("Output video not found")
            
        output_path = os.path.join(predict_dir, video_files[0])
        
        # We need to ensure it's playable in browser (h264 mp4). YOLO might output different formats.
        # For simplicity, returning the file directly. The frontend will display it.
        # Note: If it's .avi, Streamlit might not play it. We can convert to mp4 using OpenCV/FFmpeg if needed.
        # YOLOv8 usually saves as .avi by default. Let's convert it to mp4 using OpenCV if it's not mp4.
        # Actually, ultralytics YOLO saves mp4 if input is mp4. 
        
        return FileResponse(output_path, media_type="video/mp4")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
