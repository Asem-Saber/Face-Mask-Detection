# Face Mask Detection

A real-time face mask detection system that classifies whether individuals are wearing a mask **correctly**, **incorrectly**, or **not at all** — across images, videos, and live webcam streams.

Built with **YOLOv5xu** (Ultralytics), a **FastAPI** backend, and a **Streamlit** frontend.

---

## Project Structure

```
Face_Mask_Detection/
├── backend/
│   ├── main.py              # FastAPI REST API (/predict/image, /predict/video)
│   └── Dockerfile           # Backend container (Python 3.10-slim)
├── frontend/
│   └── app.py               # Streamlit UI (image / video / webcam modes)
├── dataset.py               # Dataset preparation & YOLO-format conversion
├── train.py                 # Model training script
├── inference.py             # CLI inference (image / video / webcam)
├── data.yaml                # YOLO dataset config (3 classes, train/val/test)
└── requirements.txt         # All dependencies with pinned versions
```

---


## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset & train

```bash
python dataset.py   # prepare YOLO-format data
python train.py     # train YOLOv5xu for 50 epochs on GPU
```

Trained weights will be saved to:
```
TrainingResults/MaskDetection/weights/best.pt
```

### 3. Start the backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t mask-backend ./backend
docker run -p 8000:8000 -v ${PWD}/TrainingResults:/app/weights mask-backend
```

### 4. Start the frontend

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. (Optional) CLI inference

```bash
python inference.py --model TrainingResults/MaskDetection/weights/best.pt \
                    --source <image|video|0>   \
                    --conf 0.5                 \
                    --output InferenceResults
```

Use `--source 0` for live webcam, or pass a path to an image / video file.

---

## License

This project is open-source. Feel free to use, modify, and distribute it.
