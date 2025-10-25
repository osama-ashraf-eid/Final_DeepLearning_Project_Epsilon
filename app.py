# app.py — Improved possession & pass detection for Streamlit
import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from collections import defaultdict
import gdown
from ultralytics import YOLO
import urllib.request
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

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
st.subheader("Upload Video for Analysis")
uploaded_video = st.file_uploader("Choose a football video (.mp4)", type=["mp4"])
TRACKER_FILE = "bytetrack.yaml"

# -----------------------
# Helper functions
# -----------------------
def get_average_color(frame, box):
    x1, y1, x2, y2 = box
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([0,0,0])
    return np.mean(roi.reshape(-1,3), axis=0)

def center_in_expanded_bbox(center, box, margin=8):
    x1, y1, x2, y2 = box
    x1m, y1m = int(x1 - margin), int(y1 - margin)
    x2m, y2m = int(x2 + margin), int(y2 + margin)
    return (center[0] >= x1m and center[0] <= x2m and center[1] >= y1m and center[1] <= y2m)

# -----------------------
# Parameters (tweak these for your camera/video)
# -----------------------
HOLD_FRAMES = 3               # frames required to confirm a new owner (3 works for ~25-30fps)
PROXIMITY_PX = 60             # how close (pixels) to consider "near"
SPEED_THRESHOLD_PX_PER_S = 300.0  # if ball speed > this -> likely airborne / fast pass
OVERLAP_MARGIN = 8
MIN_FRAMES_BETWEEN_PASSES = 2  # avoid counting same pass multiple times quickly

# -----------------------
# Process when video uploaded
# -----------------------
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.flush()
    video_path = tfile.name

    st.info("Analyzing video — please wait...")
    progress_text = st.empty()
    progress_bar = st.progress(0.0)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(tmp_out.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker_arg = TRACKER_FILE if os.path.exists(TRACKER_FILE) else None

    try:
        results_stream = model.track(
            source=video_path, conf=0.4, iou=0.5,
            tracker=tracker_arg, persist=True, stream=True
        )
    except Exception as e:
        cap.release(); out.release()
        st.error(f"model.track failed: {e}")
        st.stop()

    # --- state variables for possession & passes ---
    prev_ball_center = None
    prev_ball_time = None

    candidate_id = None
    candidate_streak = 0

    confirmed_owner = None         # currently confirmed owner (player id)
    confirmed_since = 0            # frames since confirmed

    last_confirmed_owner = None    # previous confirmed owner (for pass counting)
    last_pass_frame = -9999

    possession_counter = defaultdict(int)
    team_possession_counter = defaultdict(int)
    passes = []                    # list of (from_id, to_id, frame_idx)
    team_colors = {}
    team_passes_counter = defaultdict(int)

    last_player_boxes = {}         # last seen box for each player id (for arrow drawing)
    seen_players = set()

    processed_frames = 0

    # Iterate streaming results
    for frame_idx, frame_data in enumerate(results_stream, start=1):
        try:
            frame = frame_data.orig_img.copy()
        except Exception:
            ret, frame = cap.read()
            if not ret:
                break

        processed_frames += 1
        if frame_count > 0:
            progress = min(processed_frames / frame_count, 1.0)
            progress_bar.progress(progress)
            progress_text.text(f"Processed {processed_frames}/{frame_count} frames")
        else:
            progress_text.text(f"Processed {processed_frames} frames")

        # get boxes/classes/ids
        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int) if hasattr(frame_data.boxes, "id") else np.array([])

        balls = []
        players = []

        for box, cls, tid in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            if cls == 0:  # ball
                balls.append((tid, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            elif cls in [1,2]:  # player/goalkeeper
                avg_color = get_average_color(frame, (x1,y1,x2,y2))
                team_color = assign_team(tid, avg_color, team_colors)
                team_name = "Team A" if np.mean(team_color) < 128 else "Team B"
                players.append((tid, (x1,y1,x2,y2), team_name))
                last_player_boxes[tid] = (x1,y1,x2,y2)
                seen_players.add(tid)
                # draw player box
                color = (0,0,255) if team_name=="Team A" else (255,0,0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"{team_name} #{tid}", (x1, y1-8), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

        # --- compute ball center & speed ---
        ball_center = None
        ball_speed = 0.0
        ball_box_h = 0
        if len(balls) > 0:
            bx1, by1, bx2, by2 = balls[0][1]
            ball_center = np.array([(bx1+bx2)/2.0, (by1+by2)/2.0])
            ball_box_h = (by2 - by1)
            if prev_ball_center is not None and fps > 0:
                dist = np.linalg.norm(ball_center - prev_ball_center)
                ball_speed = dist * fps
            prev_ball_center = ball_center.copy()

        # --- candidate selection: nearest player & overlap check ---
        nearest_player = None
        nearest_team = None
        nearest_box = None
        min_dist = 1e9

        if ball_center is not None and len(players) > 0:
            for pid, box, team in players:
                px1, py1, px2, py2 = box
                player_center = np.array([(px1+px2)/2.0, (py1+py2)/2.0])
                d = np.linalg.norm(ball_center - player_center)
                if d < min_dist:
                    min_dist = d
                    nearest_player = pid
                    nearest_team = team
                    nearest_box = box

        # decide if ball is likely in-air
        ball_in_air = False
        if ball_center is None:
            ball_in_air = True
        else:
            if ball_speed > SPEED_THRESHOLD_PX_PER_S:
                ball_in_air = True
            # also if bbox very small (far/air) treat as in-air
            if ball_box_h < 6:
                ball_in_air = True

        # Candidate logic: require proximity AND (not in-air OR overlap)
        candidate_ok = False
        if nearest_player is not None:
            overlap = center_in_expanded_bbox(ball_center, nearest_box, OVERLAP_MARGIN) if nearest_box is not None else False
            prox = (min_dist < PROXIMITY_PX)
            if prox and (not ball_in_air or overlap):
                candidate_ok = True

        # Update candidate streak
        if candidate_ok:
            if candidate_id == nearest_player:
                candidate_streak += 1
            else:
                candidate_id = nearest_player
                candidate_streak = 1
        else:
            candidate_id = None
            candidate_streak = 0

        # Confirm ownership only when candidate streak >= HOLD_FRAMES
        if candidate_id is not None and candidate_streak >= HOLD_FRAMES:
            new_confirmed = candidate_id
        else:
            new_confirmed = None

        # If we have a confirmed owner: update counters and passes
        if new_confirmed is not None:
            confirmed_owner = new_confirmed
            confirmed_since += 1
            # increment possession frames
            possession_counter[confirmed_owner] += 1
            team = next((t for pid,t in [(p[0],p[2]) for p in players] if pid==confirmed_owner), None)
            if team is not None:
                team_possession_counter[team] += 1

            # check pass: if last_confirmed exists and different and enough frames since last pass
            if (last_confirmed_owner is not None) and (confirmed_owner != last_confirmed_owner):
                if (frame_idx - last_pass_frame) >= MIN_FRAMES_BETWEEN_PASSES:
                    passes.append((last_confirmed_owner, confirmed_owner, frame_idx))
                    # increment team-level pass counter for receiver's team
                    recv_team = None
                    for pid,box,tname in players:
                        if pid == confirmed_owner:
                            recv_team = tname; break
                    if recv_team is not None:
                        team_passes_counter[recv_team] += 1
                    last_pass_frame = frame_idx
            # update last_confirmed_owner if different
            last_confirmed_owner = confirmed_owner
        else:
            # no confirmed owner this frame: do not change last_confirmed_owner
            confirmed_since = 0

        # Draw arrow for passes that occurred in this frame (use last seen boxes)
        # We stored passes with frame_idx; draw arrow if frame_idx just equals now (or a small window)
        recent_passes = [p for p in passes if p[2] == frame_idx]
        for pf, pt, fidx in recent_passes:
            if (pf in last_player_boxes) and (pt in last_player_boxes):
                fx1,fy1,fx2,fy2 = last_player_boxes[pf]
                tx1,ty1,tx2,ty2 = last_player_boxes[pt]
                from_center = (int((fx1+fx2)/2), int((fy1+fy2)/2))
                to_center = (int((tx1+tx2)/2), int((ty1+ty2)/2))
                cv2.arrowedLine(frame, from_center, to_center, (0,255,255), 3, tipLength=0.25)

        # Highlight current confirmed owner (if any)
        if confirmed_owner is not None:
            box = last_player_boxes.get(confirmed_owner, None)
            if box is not None:
                px1,py1,px2,py2 = box
                cv2.rectangle(frame, (px1,py1), (px2,py2), color_possession, 3)
                cv2.putText(frame, f"HAS BALL #{confirmed_owner}", (px1, py1-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_possession, 2)

        # Overlay small stats on frame
        start_y = 30
        for idx, (pid, cnt) in enumerate(possession_counter.items()):
            cv2.putText(frame, f"P{pid} Poss:{cnt}f", (10, start_y + idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)
        off = start_y + len(possession_counter)*20 + 10
        for tname, cnt in team_possession_counter.items():
            cv2.putText(frame, f"{tname}: {cnt}f", (10, off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            off += 20
        cv2.putText(frame, f"Passes: {len(passes)}", (10, off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        out.write(frame)

    # release
    cap.release()
    out.release()
    progress_bar.progress(1.0)
    progress_text.text("Processing completed")

    st.success("Analysis completed.")
    st.video(tmp_out.name)
    with open(tmp_out.name, "rb") as f:
        st.download_button("Download processed video", data=f, file_name="football_tracking_result.mp4")

    # -----------------------
    # Show textual stats & pass graph
    # -----------------------
    st.write("### Ball Possession Summary (frames)")
    for pid, cnt in possession_counter.items():
        st.write(f"Player {pid}: {cnt} frames")

    st.write("### Team Possession (frames)")
    for tname, cnt in team_possession_counter.items():
        st.write(f"{tname}: {cnt} frames")

    st.write("### Pass list (from -> to) and frame")
    for i, (a,b,fidx) in enumerate(passes, start=1):
        st.write(f"{i}. Player {a} → Player {b}  (frame {fidx})")

    # pass counts matrix
    if len(passes) > 0:
        pc = defaultdict(int)
        for a,b,_ in passes:
            pc[(a,b)] += 1
        rows = [{"from":a,"to":b,"count":c} for (a,b),c in pc.items()]
        df = pd.DataFrame(rows).sort_values("count", ascending=False)
        st.write("### Pass counts table")
        st.dataframe(df, use_container_width=True)

        # draw pass graph
        G = nx.DiGraph()
        for p in seen_players:
            G.add_node(p)
        for (a,b),c in pc.items():
            G.add_edge(a,b,weight=c)
        plt.figure(figsize=(8,6))
        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u,v in G.edges()]
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800,
                arrows=True, width=np.array(weights)*0.5, edge_color='gray', font_size=9)
        edge_labels = {(u,v):d['weight'] for u,v,d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        st.pyplot(plt)
    else:
        st.info("No passes detected in this video.")

else:
    st.info("Upload a .mp4 football video to begin analysis.")
