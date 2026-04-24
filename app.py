import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(page_title="RoadGuardian Live", layout="wide")
st.title("🚧 RoadGuardian: Multi-Model Live Analytics")
st.markdown("Monitoring infrastructure for **potholes** and **trash**.")

# --- 1. Model Loading (Cached) ---
@st.cache_resource
def load_models():
    # Update these paths to your actual .tflite or .pt files
    p_model = YOLO("models/best_int8.tflite", task="detect")
    t_model = YOLO("models/yoloe-11l-seg_float16.tflite", task="detect")
    return p_model, t_model

p_model, t_model = load_models()

# --- 2. Helper: Detection & Matrix Logic ---
def extract_detection_matrix(results, label_name, frame_area):
    """Parses YOLO results into a structured list for the matrix."""
    matrix_data = []
    for r in results:
        for box in r.boxes:
            # Get coordinates as a list [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = coords
            
            # Calculate Severity based on area ratio
            obj_area = (x2 - x1) * (y2 - y1)
            ratio = (obj_area / frame_area) * 100
            
            if ratio > 15: severity = "🔴 High"
            elif ratio > 5: severity = "🟡 Medium"
            else: severity = "🟢 Low"
            
            matrix_data.append({
                "Type": label_name,
                "Severity": severity,
                "Confidence": round(float(box.conf), 2),
                "Coordinates": [round(c, 1) for c in coords] # Matrix format
            })
    return matrix_data

# --- 3. Sidebar Configuration ---
st.sidebar.header("Optimization Settings")
skip_frames = st.sidebar.slider("Inference Interval (Frames)", 1, 10, 5)
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
run_webcam = st.checkbox("Start RoadGuardian Feed", value=False)

# --- 4. UI Layout ---
col_vid, col_mat = st.columns([2, 1])
FRAME_WINDOW = col_vid.image([])
MATRIX_WINDOW = col_mat.empty()

# --- 5. Main Execution Loop ---
if run_webcam:
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    # Placeholders for persistent data across skipped frames
    last_p_res = None
    last_t_res = None
    last_matrix = []

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not detected.")
            break

        frame_count += 1
        h, w, _ = frame.shape
        frame_area = h * w

        # --- HEAVY AI INFERENCE (Every X frames) ---
        if frame_count % skip_frames == 0:
            last_p_res = p_model.predict(frame, conf=conf_thresh, verbose=False)
            last_t_res = t_model.predict(frame, conf=conf_thresh, verbose=False)
            
            # Rebuild the matrix data
            p_data = extract_detection_matrix(last_p_res, "Pothole", frame_area)
            t_data = extract_detection_matrix(last_t_res, "Trash", frame_area)
            last_matrix = p_data + t_data
            
            # Avoid counter overflow
            if frame_count > 1000: frame_count = 0

        # --- RENDERING & VISUALIZATION ---
        if last_p_res is not None and last_t_res is not None:
            # Plot Pothole boxes using built-in method
            annotated_frame = last_p_res[0].plot()
            
            # Manually overlay Trash boxes from the second model
            for box in last_t_res[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Green box for trash
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Trash {box.conf[0]:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert BGR (OpenCV) to RGB (Streamlit)
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(display_frame)
            
            # Update Data Matrix on UI
            if last_matrix:
                MATRIX_WINDOW.table(pd.DataFrame(last_matrix))
            else:
                MATRIX_WINDOW.write("Scanning for issues...")
        else:
            # Fallback for the very first few frames before inference starts
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(raw_rgb)

    cap.release()
else:
    st.info("Check the box in the sidebar to begin the live detection feed.")
