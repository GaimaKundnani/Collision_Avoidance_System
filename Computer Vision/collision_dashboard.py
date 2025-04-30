import streamlit as st
import cv2
from ultralytics import YOLO
import pyttsx3
import tempfile
import time
from datetime import datetime
import pandas as pd
import os

# Initialize model and TTS
model = YOLO("yolov8n.pt")
engine = pyttsx3.init()

# Globals
log_data = []
alert_enabled = True
snapshot_counter = 1
danger_count = 0  # Tracks the number of DANGER alerts
last_frame = None  # Store last frame for snapshot

# Streamlit setup
st.set_page_config(page_title="Collision Detection Dashboard", layout="wide")
st.title("🚗 Smart Collision Avoidance System Dashboard")
st.sidebar.title("🔧 Settings")
alert_enabled = st.sidebar.checkbox("Enable Audio Alert", value=True)
st.sidebar.write("Upload a video or use webcam")

# Upload video or use webcam
video_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi"])
use_webcam = st.sidebar.checkbox("Use Webcam")

# Add live time display
st.sidebar.markdown(f"🕒 **Current Time:** {datetime.now().strftime('%H:%M:%S')}")

# Main panel UI
status_placeholder = st.empty()
frame_display = st.empty()
log_table = st.empty()

# Video capture logic
if use_webcam:
    cap = cv2.VideoCapture(0)
elif video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
else:
    st.warning("Please upload a video or enable webcam.")
    st.stop()

# Filter for logs
status_filter = st.sidebar.selectbox("Filter Logs By Status", ["All", "SAFE", "CAUTION", "DANGER"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)[0]

    alert_triggered = False
    current_status = "SAFE"
    current_color = "green"
    current_time = datetime.now().strftime("%H:%M:%S")

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        width = x2 - x1
        distance = int(1000 / width)

        if width > 250:
            color = (0, 0, 255)
            status = "DANGER"
            alert_triggered = True
            current_status = "DANGER"
            current_color = "red"
            danger_count += 1  # Increment danger count
        elif 150 < width <= 250:
            color = (0, 165, 255)
            status = "CAUTION"
            current_status = "CAUTION"
            current_color = "orange"
        else:
            color = (0, 255, 0)
            status = "SAFE"
            current_status = "SAFE"
            current_color = "green"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} - {status}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Dist: {distance}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        log_data.append({
            "Time": current_time,
            "Object": label,
            "Distance": distance,
            "Status": status
        })

    if alert_triggered and alert_enabled:
        engine.say("Warning! Object very close!")
        engine.runAndWait()

    # Convert for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    last_frame = frame_rgb.copy()  # Store current frame
    frame_display.image(frame_rgb, channels="RGB")

    # Status and logs
    status_placeholder.markdown(
        f"<h3 style='color:{current_color};'>Current Status: {current_status}</h3>",
        unsafe_allow_html=True
    )

    # Display top 3 closest objects
    if log_data:
        df_top = pd.DataFrame(log_data).sort_values("Distance").head(3)
        st.subheader("📊 Top 3 Closest Objects")
        st.dataframe(df_top, use_container_width=True)

    # Filtered logs display
    filtered_logs = pd.DataFrame(log_data[-50:])
    if status_filter != "All":
        filtered_logs = filtered_logs[filtered_logs["Status"] == status_filter]
    log_table.dataframe(filtered_logs, use_container_width=True)

    # Danger count display
    st.sidebar.metric("🚨 Total DANGER Alerts", danger_count)

    time.sleep(0.03)

cap.release()
st.success("✅ Video processing finished.")
