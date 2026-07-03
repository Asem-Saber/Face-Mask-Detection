import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import tempfile

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="Face Mask Detection", layout="wide")
st.title("Face Mask Detection")

mode = st.sidebar.selectbox("Select Mode", ["Image Upload", "Video Upload", "Real-Time Webcam"])

if mode == "Image Upload":
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("Detection Result")
            with st.spinner("Processing..."):
                # Send to backend
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/predict/image", files=files)
                
                if response.status_code == 200:
                    result_image = Image.open(io.BytesIO(response.content))
                    st.image(result_image, use_container_width=True)
                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")

elif mode == "Video Upload":
    st.header("Video Upload")
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Video")
            # Save original to a temporary file to display
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            st.video(tfile.name)
            
        with col2:
            st.subheader("Detection Result")
            if st.button("Process Video"):
                with st.spinner("Processing... This may take a while depending on video length."):
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/predict/video", files=files)
                    
                    if response.status_code == 200:
                        # Save returned video to display
                        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        out_file.write(response.content)
                        out_file.close()
                        st.video(out_file.name)
                    else:
                        st.error(f"Error from backend: {response.status_code} - {response.text}")

elif mode == "Real-Time Webcam":
    st.header("Real-Time Webcam Detection")
    st.markdown("Uses `streamlit-webrtc` to stream your webcam. Frames are processed using the backend.")
    
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Send frame to backend
            _, encoded_img = cv2.imencode('.jpg', img)
            files = {"file": ("frame.jpg", encoded_img.tobytes(), "image/jpeg")}
            try:
                response = requests.post(f"{BACKEND_URL}/predict/image", files=files)
                if response.status_code == 200:
                    # Decode result
                    nparr = np.frombuffer(response.content, np.uint8)
                    result_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    return result_img
                else:
                    return img
            except Exception as e:
                print("Error sending frame:", e)
                return img

    webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor, 
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
