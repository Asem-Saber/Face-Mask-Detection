import os
import subprocess
import cv2
import numpy as np
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, FileResponse
from ultralytics import YOLO
import imageio_ffmpeg

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI(title="Face Mask Detection API")

# Load the model
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "TrainingResults", "MaskDetection", "weights", "best.pt"))
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
    temp_base = tempfile.gettempdir()
    input_path = os.path.join(temp_base, f"{temp_id}_input.mp4")
    raw_path = os.path.join(temp_base, f"{temp_id}_raw.avi")
    output_path = os.path.join(temp_base, f"{temp_id}_output.mp4")

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, conf=0.5, verbose=False)
            writer.write(results[0].plot())

        cap.release()
        writer.release()

        subprocess.run([
            FFMPEG_PATH, "-y", "-i", raw_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path
        ], check=True, capture_output=True)

        return FileResponse(output_path, media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
