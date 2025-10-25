import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
import gdown
import urllib.request
from PIL import Image

# -----------------------
# إعداد ملف التتبع
# -----------------------
if not os.path.exists("bytetrack.yaml"):
    tracker_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/bytetrack.yaml"
    urllib.request.urlretrieve(tracker_url, "bytetrack.yaml")

# -----------------------
# إعداد واجهة التطبيق
# -----------------------
st.set_page_config(page_title="Football Tracking", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0b7dda;'>⚽ Football Tracking</h1>", unsafe_allow_html=True)

# عرض الصورة في واجهة التطبيق
DEFAULT_IMAGE = "football_img.jpg"
if os.path.exists(DEFAULT_IMAGE):
    st.image(DEFAULT_IMAGE, use_container_width=True, caption="Automated Football Match Tracking")
else:
    st.info("⚠️ Default image not found — please add 'football_img.jpg' to your app directory.")

st.write("---")

# -----------------------
# تحميل الموديل
# -----------------------
MODEL_PATH = "yolov8m-football_ball_only.pt"
DRIVE_FILE_ID = "1jDmtelt3wJgxRj0j7928VyFNAjq0dzk4"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading YOLO model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# -----------------------
# رفع الفيديو
# -----------------------
st.subheader("🎥 Upload Football Video")
uploaded_video = st.file_uploader("Upload a football video (.mp4)", type=["mp4"])
TRACKER_FILE = "bytetrack.yaml"

# -----------------------
# ألوان العناصر
# -----------------------
color_ball = (0, 255, 255)        # Yellow for ball
color_possession = (0, 255, 0)    # Green for player with ball
color_referee = (200, 200, 200)   # Grey for referee

# -----------------------
# دوال مساعدة
# -----------------------
def get_average_color(frame, box):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
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
                team_colors[player_id] = team_colors[assigned_team]
            else:
                team_colors[player_id] = color
    return team_colors[player_id]

# -----------------------
# تنفيذ التتبع
# -----------------------
if uploaded_video is not None:
    # حفظ الفيديو مؤقتًا
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.info("Processing video... Please wait ⏳")
    progress_bar = st.progress(0.0)
    progress_text = st.empty()

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker_arg = TRACKER_FILE if os.path.exists(TRACKER_FILE) else None
    results_stream = model.track(source=video_path, conf=0.4, iou=0.5, tracker=tracker_arg, persist=True, stream=True)

    team_colors = {}
    processed_frames = 0

    for frame_data in results_stream:
        try:
            frame = frame_data.orig_img.copy()
        except:
            ret, frame = cap.read()
            if not ret:
                break

        processed_frames += 1
        if frame_count > 0:
            progress = processed_frames / frame_count
            progress_bar.progress(min(progress, 1.0))
        progress_text.text(f"Processing frame {processed_frames}/{frame_count}")

        box_ids = getattr(frame_data.boxes, "id", None)
        if box_ids is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        balls, players = [], []

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            if cls == 0:  # Ball
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

            elif cls in [1, 2]:  # Player / Goalkeeper
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                team_color = assign_team(track_id, avg_color, team_colors)

                if np.mean(team_color) < 128:
                    draw_color = (0, 0, 255)
                    team_name = "Team A"
                else:
                    draw_color = (255, 0, 0)
                    team_name = "Team B"

                players.append((track_id, (x1, y1, x2, y2), team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

            else:  # Referee
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

        # تحديد اللاعب اللي معاه الكورة
        current_owner_id = None
        if len(balls) > 0 and len(players) > 0:
            bx1, by1, bx2, by2 = balls[0][1]
            ball_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
            min_dist = 1e9
            for player_id, box, team_name in players:
                px1, py1, px2, py2 = box
                player_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
                dist = np.linalg.norm(ball_center - player_center)
                if dist < min_dist:
                    min_dist = dist
                    current_owner_id = player_id

            if min_dist < 90 and current_owner_id is not None:
                for player_id, box, team_name in players:
                    if player_id == current_owner_id:
                        px1, py1, px2, py2 = box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                        cv2.putText(frame, f"{team_name} #{player_id} HAS THE BALL",
                                    (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        out.write(frame)

    cap.release()
    out.release()
    progress_text.text("✅ Processing completed!")
    progress_bar.progress(1.0)

    st.success("🎬 Video processed successfully!")
    st.video(tmp_out.name)
    with open(tmp_out.name, "rb") as f:
        st.download_button("📥 Download Processed Video", data=f, file_name="football_tracking_result.mp4")

else:
    st.info("📤 Please upload a .mp4 football video to start tracking.")
