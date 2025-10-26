import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "1"

# ===============================
# التأكد من وجود OpenCV
# ===============================
try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "opencv-python-headless==4.9.0.80"])
    import cv2

import streamlit as st
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from tempfile import NamedTemporaryFile
import gdown

# ===============================
# إعداد صفحة Streamlit
# ===============================
st.set_page_config(page_title="⚽ Detection and Tracking", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #0a84ff;'>⚽ Detection and Tracking</h1>
    <div style='text-align:center;'>
        <img src='https://images.unsplash.com/photo-1517927033932-b3d18e61fb3a'
             style='width:100%; border-radius: 12px; margin-bottom: 30px;'>
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# تحميل الموديل من Google Drive لو مش موجود
# ===============================
MODEL_PATH = "yolov8m-football_ball_only.pt"
DRIVE_URL = "https://drive.google.com/uc?id=1jDmtelt3wJgxRj0j7928VyFNAjq0dzk4"

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading YOLO model from Google Drive...")
    try:
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# ===============================
# تحميل الموديل
# ===============================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading YOLO model: {e}")
    st.stop()

# ===============================
# واجهة المستخدم
# ===============================
video_file = st.file_uploader("🎬 Upload a football video", type=["mp4", "mov", "avi"])

if video_file:
    tfile = NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_path)
    st.write("🏃 Running tracking... This may take a while depending on video length.")

    tracker_config = "botsort.yaml"

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    color_player = (0, 150, 255)
    color_goalkeeper = (255, 255, 0)
    color_ball = (0, 255, 255)
    color_referee = (200, 200, 200)
    color_possession = (0, 255, 0)

    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors = {}
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    def get_average_color(frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([0, 0, 0])
        return np.mean(roi.reshape(-1, 3), axis=0)

    def assign_team(player_id, color):
        if player_id not in team_colors:
            if len(team_colors) == 0:
                team
