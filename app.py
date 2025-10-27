import os
# import subprocess # Removed: not needed for runtime installation
# import sys # Removed: not needed for runtime installation

# ===============================
# Fix OpenCV and libGL Issues (Streamlit Cloud)
# We rely on packages.txt and requirements.txt for installation,
# but keep the environment variables to ensure proper video handling.
# ===============================
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "1"

# We assume a successful import, as dependencies are now configured externally.
import cv2
cv2.setNumThreads(1)

import streamlit as st
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import tempfile
import gdown

# ===============================
# Streamlit UI Setup
# ===============================
st.set_page_config(page_title="⚽ Detection and Tracking", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #0a84ff;'>⚽ Detection and Tracking</h1>
<div style='text-align:center;'>
    <img src='https://images.unsplash.com/photo-1517927033932-b3d18e61fb3a'
             style='width:100%; border-radius: 12px; margin-bottom: 30px;'>
</div>
""", unsafe_allow_html=True)

# ===============================
# YOLO Model Download
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
# Load YOLO Model
# ===============================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading YOLO model: {e}")
    st.stop()

# ===============================
# File Upload Section
# ===============================
video_file = st.file_uploader("🎬 Upload a football video", type=["mp4", "mov", "avi"])

if video_file:
    fd, video_path = tempfile.mkstemp(suffix=".mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.video(video_path)
    st.write("🏃 Running tracking... Please wait ⏳")

    tracker_config = "botsort.yaml"
    if not os.path.exists(tracker_config):
        with open(tracker_config, "w") as f:
            # FIX: Added comprehensive tracker parameters required by ByteTrack implementation
            f.write("tracker_type: bytetrack\n")
            f.write("track_buffer: 30\n")
            f.write("track_high_thresh: 0.5\n")
            f.write("track_low_thresh: 0.1\n")
            f.write("new_track_thresh: 0.7\n")
            f.write("match_thresh: 0.8\n")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    output_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    color_ball = (0, 255, 255)
    color_possession = (0, 255, 0)
    color_referee = (200, 200, 200)

    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors = {}
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    # ===============================
    # Helper Functions
    # ===============================
    def get_average_color(frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([0, 0, 0])
        return np.mean(roi.reshape(-1, 3), axis=0)

    def assign_team(player_id, color):
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
                    team_colors[player_id] = team_colors[assigned_team]
                else:
                    team_colors[player_id] = color
        return team_colors[player_id]

    # ===============================
    # Run YOLO Tracking
    # ===============================
    results = model.track(
        source=video_path,
        conf=0.4,
        iou=0.5,
        tracker=tracker_config,
        persist=True,
        stream=True
    )

    for frame_data in results:
        frame = frame_data.orig_img.copy()
        if frame_data.boxes.id is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        balls, players = [], []
        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            if cls == 0:
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)
            elif cls in [1, 2]:
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                team_color = assign_team(track_id, avg_color)
                draw_color = (0, 0, 255) if np.mean(team_color) < 128 else (255, 0, 0)
                team_name = "Team A" if np.mean(team_color) < 128 else "Team B"
                players.append((track_id, (x1, y1, x2, y2), team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{track_id}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

        current_owner_id, current_owner_team = None, None
        if balls and players:
            bx1, by1, bx2, by2 = balls[0][1]
            ball_center = np.array([(bx1 + bx2)/2, (by1 + by2)/2])
            min_dist = 1e9
            for player_id, box, team_name in players:
                px1, py1, px2, py2 = box
                player_center = np.array([(px1 + px2)/2, (py1 + py2)/2])
                dist = np.linalg.norm(ball_center - player_center)
                if dist < min_dist:
                    min_dist = dist
                    current_owner_id, current_owner_team = player_id, team_name
            if min_dist < 90:
                possession_counter[current_owner_id] += 1
                team_possession_counter[current_owner_team] += 1
                if last_owner_id is not None and current_owner_id != last_owner_id:
                    passes.append((last_owner_id, current_owner_id))
                    team_passes_counter[current_owner_team] += 1
                last_owner_id = current_owner_id

        if current_owner_id is not None:
            for player_id, box, team_name in players:
                if player_id == current_owner_id:
                    px1, py1, px2, py2 = box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                    cv2.putText(frame, f"{team_name} #{player_id} HAS THE BALL",
                                (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Tracking completed!")
    st.video(output_path)

    st.subheader("📊 Ball Possession Summary - Players")
    for player_id, count in possession_counter.items():
        st.write(f"Player {player_id}: {count} frames")

    st.subheader("🏳️ Ball Possession Summary - Teams")
    for team_name, count in team_possession_counter.items():
        st.write(f"{team_name}: {count} frames")

    st.subheader("🔁 Total Passes")
    for i, (from_id, to_id) in enumerate(passes, 1):
        st.write(f"{i}. Player {from_id} → Player {to_id}")

    st.subheader("📈 Passes per Team")
    for team_name, count in team_passes_counter.items():
        st.write(f"{team_name}: {count} passes")

else:
    st.info("👆 Please upload a video to start detection.")
