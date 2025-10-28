import os
import cv2
import streamlit as st
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import tempfile
import gdown

# ===============================
# Streamlit إعداد الواجهة
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
# تحميل الموديل من Google Drive
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
# تحميل نموذج YOLO
# ===============================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading YOLO model: {e}")
    st.stop()

# ===============================
# رفع الفيديو
# ===============================
video_file = st.file_uploader("🎬 Upload a football video", type=["mp4", "mov", "avi"])

if video_file:
    fd, video_path = tempfile.mkstemp(suffix=".mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.video(video_path)
    st.write("🏃 Running detection... Please wait ⏳")

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
    # دوال مساعدة
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
    # تنفيذ الكشف (بدون tracker)
    # ===============================
    results = model.predict(
        source=video_path,
        conf=0.4,
        iou=0.5,
        stream=True
    )

    for frame_data in results:
        frame = frame_data.orig_img.copy()
        if not hasattr(frame_data, "boxes") or frame_data.boxes is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)

        balls, players = [], []
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            if cls == 0:  # Ball
                balls.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)
            elif cls in [1, 2]:  # Players
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                draw_color = (0, 0, 255) if np.mean(avg_color) < 128 else (255, 0, 0)
                team_name = "Team A" if np.mean(avg_color) < 128 else "Team B"
                players.append((x1, y1, x2, y2, team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)
            else:  # Referee
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

        out.write(frame)

    cap.release()
    out.release()

    # ===============================
    # عرض النتائج
    # ===============================
    st.success("✅ Detection completed!")
    st.video(output_path)

else:
    st.info("👆 Please upload a video to start detection.")
