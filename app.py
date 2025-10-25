import streamlit as st
import os
import tempfile
import cv2
import numpy as np
import gdown
import urllib.request
from ultralytics import YOLO

# -----------------------
# Ensure tracker config exists
# -----------------------
if not os.path.exists("bytetrack.yaml"):
    tracker_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/bytetrack.yaml"
    urllib.request.urlretrieve(tracker_url, "bytetrack.yaml")

# -----------------------
# App config (UI text in English)
# -----------------------
st.set_page_config(page_title="Football Tracking & Analysis", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0b7dda;'>⚽ Football Tracking & Analysis</h1>", unsafe_allow_html=True)

DEFAULT_IMAGE = "football_img.jpg"
if os.path.exists(DEFAULT_IMAGE):
    st.image(DEFAULT_IMAGE, use_container_width=True, caption="Automated Football Match Analysis")
else:
    st.info("Default header image not found in repo. Add 'football_img.jpg' to repository root if you want a header image.")

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
st.subheader("Upload Video for Tracking")
uploaded_video = st.file_uploader("Choose a football video (.mp4)", type=["mp4"])
TRACKER_FILE = "bytetrack.yaml"

# -----------------------
# Helper functions
# -----------------------
def get_average_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([0, 0, 0])
    return np.mean(roi.reshape(-1, 3), axis=0)

def assign_team_color(player_id, color, team_colors):
    if player_id not in team_colors:
        if len(team_colors) == 0:
            team_colors[player_id] = color
        else:
            distances = [np.linalg.norm(color - c) for c in team_colors.values()]
            if min(distances) < 40:
                team_colors[player_id] = list(team_colors.values())[0]
            else:
                team_colors[player_id] = color
    return team_colors[player_id]

# -----------------------
# Process video
# -----------------------
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.info("Tracking in progress... Please wait ⏳")
    progress_bar = st.progress(0.0)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (w, h))

    tracker_arg = TRACKER_FILE if os.path.exists(TRACKER_FILE) else None

    team_colors = {}
    last_owner_id = None
    NEAR_DISTANCE = 70  # pixel distance to decide if a player owns the ball

    try:
        results_stream = model.track(
            source=video_path,
            conf=0.4,
            iou=0.5,
            tracker=tracker_arg,
            persist=True,
            stream=True
        )
    except Exception as e:
        st.error(f"Tracking failed: {e}")
        st.stop()

    frame_idx = 0
    color_ball = (0, 255, 255)
    color_possession = (0, 255, 0)

    for frame_data in results_stream:
        frame = frame_data.orig_img.copy()
        frame_idx += 1
        progress_bar.progress(min(frame_idx / frame_count, 1.0))

        ids = getattr(frame_data.boxes, "id", None)
        if ids is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        balls, players = [], []

        # --- Draw detections ---
        for box, cls, tid in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            if cls == 0:  # ball
                balls.append((tid, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ball, 2)

            elif cls in [1, 2]:  # player
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                team_color = assign_team_color(tid, avg_color, team_colors)
                if np.mean(team_color) < 128:
                    draw_color = (0, 0, 255)  # Red Team
                    team_name = "Team Red"
                else:
                    draw_color = (255, 0, 0)  # Blue Team
                    team_name = "Team Blue"

                players.append((tid, (x1, y1, x2, y2), team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

        # --- Determine who has the ball ---
        current_owner_id = None
        if len(balls) > 0 and len(players) > 0:
            bx1, by1, bx2, by2 = balls[0][1]
            ball_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])

            min_dist = 1e9
            for pid, (x1, y1, x2, y2), team_name in players:
                player_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                dist = np.linalg.norm(ball_center - player_center)
                if dist < min_dist:
                    min_dist = dist
                    current_owner_id = pid

            if min_dist < NEAR_DISTANCE:
                last_owner_id = current_owner_id

        # Highlight player who has the ball
        if last_owner_id is not None:
            for pid, (x1, y1, x2, y2), team_name in players:
                if pid == last_owner_id:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_possession, 4)
                    cv2.putText(frame, f"{team_name} #{pid} HAS THE BALL",
                                (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_possession, 2)

        out.write(frame)

    cap.release()
    out.release()
    progress_bar.progress(1.0)
    st.success("Tracking completed successfully ✅")

    st.video(tmp_out.name)
    with open(tmp_out.name, "rb") as f:
        st.download_button("Download processed video", f, "football_tracking_result.mp4")

else:
    st.info("Upload a .mp4 football video to begin tracking.")
