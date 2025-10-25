import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from collections import defaultdict
import gdown
from ultralytics import YOLO
from PIL import Image

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="Football Tracking & Analysis", layout="wide")
st.markdown("<h1 style='text-align:center;color:#0b7dda;'>⚽ Football Tracking & Analysis</h1>", unsafe_allow_html=True)

# Show header image (from repo)
DEFAULT_IMAGE = "football_img.jpg"
if os.path.exists(DEFAULT_IMAGE):
    st.image(DEFAULT_IMAGE, use_column_width=True, caption="Automated Football Match Analysis")
else:
    st.info("Default header image not found in repo. Upload or add 'football_img.jpg' to the repository root.")

st.write("---")

# -----------------------
# Model download (Google Drive)
# -----------------------
MODEL_PATH = "yolov8m-football_ball_only.pt"
DRIVE_FILE_ID = "1jDmtelt3wJgxRj0j7928VyFNAjq0dzk4"  # your file id
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
st.write("Upload an MP4 video. After processing, the annotated video will be shown and available for download.")
uploaded_video = st.file_uploader("Choose a football video (.mp4)", type=["mp4"])

# Optional tracker file: botsort.yaml (if not present, model.track will try to run without it)
TRACKER_FILE = "botsort.yaml"
if not os.path.exists(TRACKER_FILE):
    st.info("Optional tracker file 'botsort.yaml' not found in repo. Tracking will continue using default settings if available.")

# -----------------------
# Helper functions / logic (from your original script)
# -----------------------
color_ball = (0, 255, 255)
color_referee = (200, 200, 200)
color_possession = (0, 255, 0)

def get_average_color(frame, box):
    x1, y1, x2, y2 = box
    # clamp
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
                team_colors[player_id] = team_colors[assigned_team]
            else:
                team_colors[player_id] = color
    return team_colors[player_id]

# -----------------------
# Process video when uploaded
# -----------------------
if uploaded_video is not None:
    # Save uploaded file to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()
    video_path = tfile.name

    st.info("Starting analysis. This will run on the server — please be patient.")
    progress_text = st.empty()
    progress_bar = st.progress(0.0)

    # Open video and prepare writer
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (w, h))

    # Prepare tracking using model.track (stream=True)
    tracker_arg = TRACKER_FILE if os.path.exists(TRACKER_FILE) else None

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
        cap.release()
        out.release()
        st.error(f"model.track failed: {e}")
        st.stop()

    # State for possession/passes
    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors = {}
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    processed_frames = 0

    # Iterate streaming results (each frame_data corresponds to a processed frame)
    for frame_data in results_stream:
        try:
            frame = frame_data.orig_img.copy()
        except Exception:
            # fallback: try to read directly from capture if available
            ret, frame = cap.read()
            if not ret:
                break

        processed_frames += 1
        # Update progress
        if frame_count > 0:
            progress = min(processed_frames / frame_count, 1.0)
            progress_bar.progress(progress)
            progress_text.text(f"Processed {processed_frames}/{frame_count} frames")
        else:
            progress_text.text(f"Processed {processed_frames} frames")

        # If there are no tracked boxes (ids), just write frame
        box_ids = getattr(frame_data.boxes, "id", None)
        if box_ids is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        balls = []
        players = []

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            if cls == 0:  # ball
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

            elif cls in [1, 2]:  # goalkeeper or player
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
                cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

            else:  # referee or other
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

        # Ball possession logic
        current_owner_id = None
        current_owner_team = None
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
                    current_owner_team = team_name

            if min_dist < 90:
                possession_counter[current_owner_id] += 1
                team_possession_counter[current_owner_team] += 1

                if last_owner_id is not None and current_owner_id != last_owner_id:
                    passes.append((last_owner_id, current_owner_id))
                    team_passes_counter[current_owner_team] += 1
                last_owner_id = current_owner_id

        # Highlight current owner
        if current_owner_id is not None:
            for player_id, box, team_name in players:
                if player_id == current_owner_id:
                    px1, py1, px2, py2 = box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                    cv2.putText(frame, f"{team_name} #{player_id} HAS THE BALL",
                                (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        # Overlay possession & passes summary (top-left)
        start_y = 30
        for idx, (player_id, count) in enumerate(possession_counter.items()):
            cv2.putText(frame, f"P{player_id} Possession: {count} frames",
                        (10, start_y + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        offset = start_y + len(possession_counter) * 25 + 10
        for team_name, count in team_possession_counter.items():
            cv2.putText(frame, f"{team_name} Possession: {count} frames",
                        (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            offset += 25

        cv2.putText(frame, f"Total Passes: {len(passes)}", (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    progress_bar.progress(1.0)
    progress_text.text("Processing completed.")

    # Show video and provide download
    st.success("Analysis completed.")
    st.video(tmp_out.name)
    with open(tmp_out.name, "rb") as f:
        st.download_button("Download processed video", data=f, file_name="football_tracking_result.mp4")

    # Show textual stats
    st.write("### Ball Possession Summary (frames per player)")
    for player_id, count in possession_counter.items():
        st.write(f"Player {player_id}: {count} frames")

    st.write("### Team Possession (frames)")
    for team_name, count in team_possession_counter.items():
        st.write(f"{team_name}: {count} frames")

    st.write("### Total Passes")
    for i, (from_id, to_id) in enumerate(passes, 1):
        st.write(f"{i}. Player {from_id} -> Player {to_id}")

else:
    st.info("Upload a .mp4 football video to begin analysis.")

