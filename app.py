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
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------
# Ensure tracker config exists
# -----------------------
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
                team_colors[player_id] = team_colors[assigned_team]
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

    st.info("Analyzing video... Please wait ⏳")
    progress_text = st.empty()
    progress_bar = st.progress(0.0)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (w, h))

    tracker_arg = TRACKER_FILE if os.path.exists(TRACKER_FILE) else None

    try:
        results_stream = model.track(
            source=video_path, conf=0.4, iou=0.5,
            tracker=tracker_arg, persist=True, stream=True
        )
    except Exception as e:
        st.error(f"model.track failed: {e}")
        st.stop()

    # --- state ---
    last_owner_id = None                # last confirmed owner (used for counting passes)
    possession_counter = defaultdict(int)
    passes = []                         # list of tuples (from_id, to_id)
    team_colors = {}
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    # stable ownership variables
    stable_owner_id = None
    stable_count = 0
    STABLE_FRAMES = 3            # frames required to confirm a new owner
    PROXIMITY_THRESHOLD = 70    # pixels to consider "near"
    IN_AIR_HEIGHT = 8           # ball bbox height less than this -> likely in air

    processed_frames = 0

    # For pass-graph plotting later, we will accumulate set of player IDs seen and passes
    seen_players = set()

    # Iterate stream
    for frame_data in results_stream:
        try:
            frame = frame_data.orig_img.copy()
        except Exception:
            ret, frame = cap.read()
            if not ret:
                break

        processed_frames += 1
        progress = min(processed_frames / max(frame_count, 1), 1.0)
        progress_bar.progress(progress)
        progress_text.text(f"Processed {processed_frames}/{frame_count} frames")

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
            if cls == 0:  # ball
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)
            elif cls in [1, 2]:  # player
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                team_color = assign_team(track_id, avg_color, team_colors)
                if np.mean(team_color) < 128:
                    draw_color = (0, 0, 255); team_name = "Team A"
                else:
                    draw_color = (255, 0, 0); team_name = "Team B"
                players.append((track_id, (x1, y1, x2, y2), team_name))
                seen_players.add(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

        # --- Ball possession logic (stable owner) ---
        current_owner_id = None
        current_owner_team = None
        if len(balls) > 0 and len(players) > 0:
            bx1, by1, bx2, by2 = balls[0][1]
            ball_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
            ball_h = (by2 - by1)

            # detect likely in-air by small bbox height
            ball_in_air = ball_h < IN_AIR_HEIGHT

            # find nearest player
            min_dist = 1e9
            nearest_player = None
            nearest_team = None
            nearest_box = None
            for player_id, box, team_name in players:
                px1, py1, px2, py2 = box
                player_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
                dist = np.linalg.norm(ball_center - player_center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_player = player_id
                    nearest_team = team_name
                    nearest_box = box

            # only consider if ball not in air and within proximity
            if (not ball_in_air) and (min_dist < PROXIMITY_THRESHOLD):
                # stable check
                if stable_owner_id == nearest_player:
                    stable_count += 1
                else:
                    stable_owner_id = nearest_player
                    stable_count = 1

                if stable_count >= STABLE_FRAMES:
                    current_owner_id = stable_owner_id
                    current_owner_team = nearest_team

                    # update possession counters
                    possession_counter[current_owner_id] += 1
                    team_possession_counter[current_owner_team] += 1

                    # count pass only if last_owner_id exists and is different
                    if (last_owner_id is not None) and (current_owner_id != last_owner_id):
                        passes.append((last_owner_id, current_owner_id))
                        team_passes_counter[current_owner_team] += 1
                        # draw arrow between last_owner and current_owner if available in current players list
                        p_from = next((p for p in players if p[0] == last_owner_id), None)
                        p_to = next((p for p in players if p[0] == current_owner_id), None)
                        if p_from is not None and p_to is not None:
                            fx1, fy1, fx2, fy2 = p_from[0] if False else p_from[1]  # safe unpack
                            # correct unpack:
                            fx1, fy1, fx2, fy2 = p_from[1]
                            tx1, ty1, tx2, ty2 = p_to[1]
                            from_center = (int((fx1 + fx2) / 2), int((fy1 + fy2) / 2))
                            to_center = (int((tx1 + tx2) / 2), int((ty1 + ty2) / 2))
                            cv2.arrowedLine(frame, from_center, to_center, (0, 255, 255), 4, tipLength=0.3)

                    # set last owner if not set (this ensures first confirmed owner becomes last_owner for next pass)
                    if last_owner_id is None:
                        last_owner_id = current_owner_id
                    else:
                        last_owner_id = current_owner_id
            else:
                # ball in air or too far -> reset stability (but keep last_owner_id)
                stable_owner_id = None
                stable_count = 0

        # Highlight current owner (the one currently nearest; may be None)
        if current_owner_id is not None:
            for player_id, box, team_name in players:
                if player_id == current_owner_id:
                    px1, py1, px2, py2 = box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                    cv2.putText(frame, f"{team_name} #{player_id} HAS THE BALL",
                                (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        # Overlay stats
        start_y = 30
        for idx, (player_id, count) in enumerate(possession_counter.items()):
            cv2.putText(frame, f"P{player_id} Possession: {count}f",
                        (10, start_y + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        offset = start_y + len(possession_counter) * 25 + 10
        for team_name, count in team_possession_counter.items():
            cv2.putText(frame, f"{team_name}: {count}f", (10, offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            offset += 25
        cv2.putText(frame, f"Total Passes: {len(passes)}", (10, offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)

    # --- Done processing ---
    cap.release()
    out.release()
    progress_bar.progress(1.0)
    progress_text.text("Processing completed ✅")

    st.success("Analysis completed.")
    st.video(tmp_out.name)
    with open(tmp_out.name, "rb") as f:
        st.download_button("Download processed video", data=f, file_name="football_tracking_result.mp4")

    # -----------------------
    # Pass analysis + visualization
    # -----------------------
    st.write("### Ball Possession Summary")
    for player_id, count in possession_counter.items():
        st.write(f"Player {player_id}: {count} frames")

    st.write("### Team Possession")
    for team_name, count in team_possession_counter.items():
        st.write(f"{team_name}: {count} frames")

    st.write("### Total Passes")
    for i, (f_id, t_id) in enumerate(passes, 1):
        st.write(f"{i}. Player {f_id} → Player {t_id}")

    # Build pass counts matrix
    if len(passes) > 0:
        pass_counts = defaultdict(int)
        for a, b in passes:
            pass_counts[(a, b)] += 1

        # Convert to DataFrame for display
        rows = []
        for (a, b), c in pass_counts.items():
            rows.append({"from": a, "to": b, "count": c})
        df_pass = pd.DataFrame(rows).sort_values("count", ascending=False)
        st.write("### Passes table (from -> to -> count)")
        st.dataframe(df_pass, use_container_width=True)

        # Draw directed pass graph
        G = nx.DiGraph()
        for p in seen_players:
            G.add_node(p)
        for (a, b), c in pass_counts.items():
            G.add_edge(a, b, weight=c)

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800,
                arrows=True, width=np.array(weights) * 0.5, edge_color="gray", font_size=10)
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        plt.title("Pass Graph (directed)")
        st.pyplot(plt)
    else:
        st.info("No passes detected in this video.")

else:
    st.info("Upload a .mp4 football video to begin analysis.")
