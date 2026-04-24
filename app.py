import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import queue

# --- Page Config ---
st.set_page_config(page_title="RoadGuardian Cloud", layout="wide")
st.title("🚧 RoadGuardian: Multi-Model AI Monitor")

# --- 1. Model Loading (Cached) ---
@st.cache_resource
def load_models():
    # Ensure these paths match your folder structure exactly
    p_model = YOLO("models/best_int8.tflite", task="detect")
    t_model = YOLO("models/yolov8n-waste-12cls-best_int8.tflite", task="detect")
    return p_model, t_model

p_model, t_model = load_models()

# --- 2. Shared Data Queue ---
# This passes the "Matrix" data from the video thread to the UI thread
result_queue = queue.Queue()

# --- 3. Helper: Severity Logic ---
def get_severity(box, frame_area):
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    ratio = (area / frame_area) * 100
    if ratio > 15: return "🔴 High"
    if ratio > 5: return "🟡 Medium"
    return "🟢 Low"

# --- 4. Video Processing Callback ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    frame_area = h * w
    
    # Run Inference
    res_p = p_model.predict(img, conf=0.45, verbose=False)
    res_t = t_model.predict(img, conf=0.45, verbose=False)

    current_matrix = []

    # Process Potholes
    annotated_img = res_p[0].plot() # Base layer with pothole boxes
    for box in res_p[0].boxes:
        coords = box.xyxy[0].tolist()
        current_matrix.append({
            "Type": "Pothole",
            "Severity": get_severity(coords, frame_area),
            "Conf": round(float(box.conf), 2),
            "Coords": [round(c, 1) for c in coords]
        })

    # Process Trash (Manual overlay since plot() creates a new image)
    for box in res_t[0].boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, coords)
        
        # Draw on the annotated_img
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, "TRASH", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        current_matrix.append({
            "Type": "Trash",
            "Severity": get_severity(coords, frame_area),
            "Conf": round(float(box.conf), 2),
            "Coords": [round(c, 1) for c in coords]
        })

    # Push matrix to queue for the UI thread
    if current_matrix:
        result_queue.put(current_matrix)

    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- 5. UI Layout ---
col_vid, col_mat = st.columns([2, 1])

with col_vid:
    st.subheader("Live Feed")
    # STUN server allows WebRTC to work behind firewalls (like campus Wi-Fi)
    RTC_CONFIG = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="road-guardian",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_mat:
    st.subheader("Detection Matrix")
    matrix_placeholder = st.empty()
    
    # UI Refresh Loop: This runs while the webcam is active
    while ctx.state.playing:
        try:
            # Grab the latest matrix from the queue
            latest_data = result_queue.get(timeout=1)
            matrix_placeholder.table(pd.DataFrame(latest_data))
        except queue.Empty:
            matrix_placeholder.write("Scanning for infrastructure issues...")
