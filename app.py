import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from collections import defaultdict
import gdown
from ultralytics import YOLO
from PIL import Image
import urllib.request

# Ensure tracker config exists
if not os.path.exists("bytetrack.yaml"):
    tracker_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/bytetrack.yaml"
    urllib.request.urlretrieve(tracker_url, "bytetrack.yaml")

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="Football Tracking & Analysis", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0b7dda;'>⚽ Football Tracking & Analysis</h1>", unsafe_allow_html=True)

# Header image
DEFAULT_IMAGE = "football_img.jpg"
if os.path.exists(DEFAULT_IMAGE):
    st.image(DEFAULT_IMAGE, use_container_width=True, caption="Automated Football Match Analysis")
else:
    st.info("Default header image not found in repo. Upload or add 'football_img.jpg' to the repository root.")

st.write("---")

# -----------------------
# Model download (Google Drive)
# -----------------------
MODEL_PATH = "yolov8m-football_ball_only.pt"
DRIVE_FILE_ID = "1jDmtelt3wJgxRj0j7928VyFNAjq0dzk4"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Model not found locally — downloading from Google Drive (this may take a while).")
    try:
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully.")
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

# -----------------------
# Load model
# -----------------------
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# -----------------------
# Video upload UI
# -----------------------
st.subheader("Upload Video for Analysis")
uploaded_video = st.file_uploader("Choose a football video (.mp4)", type=["mp4"])

TRACKER_FILE = "bytetrack.yaml"

# -----------------------
# Helper functions
# -----------------------
color_ball = (0, 255, 255)
color_referee = (200, 200, 200)
color_possession = (0, 255, 0)

def get_average_color(frame, box):
    x1, y1, x2, y2 = box
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([0, 0, 0])
    return np.mean(roi.reshape(-1, 3), axis=0)

def assign_team(player_id, color, team_colors):
    if player_id not in team_colors:
        if len(team_colors) == 0:
            team_colors[player_id] = color
        else:
            min_dist = 1e9
            assigned_team = None
            for pid, c in team_colors.items():
                dist = np.linalg.norm(color - c)
                if dist < min_dist:
                    min_dist = dist
                    assigned_team = pid
            if min_dist < 40:
                team_colors[player_id] =_]()_
