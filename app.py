import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from collections import defaultdict
from ultralytics import YOLO

# --- Configuration and Setup ---

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

# Title and Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Football Detection and Tracking</h1>", unsafe_allow_html=True)
st.markdown("---")

# Expressive Image (Using a placeholder image URL for a football match scene)
image_url = "https://placehold.co/1200x300/171717/ffffff?text=Advanced+Football+Match+Analysis"
st.image(image_url, use_container_width=True)

st.markdown("---")
st.write("Upload a video file to run real-time player and ball tracking, team assignment, and possession analysis.")

# Model Loading (Important: Update this line with the correct path to your custom YOLO weights!)
try:
    # IMPORTANT: The model weights file 'yolov8m-football_ball_only.pt' MUST be available in the deployment environment.
    # If not available, Streamlit will fail here. Using a public model for initial setup might be safer.
    # For demonstration, we assume your custom weights are accessible.
    model = YOLO("yolov8m-football_ball_only.pt")
    # For a fully public, downloadable model, you could use: model = YOLO("yolov8n.pt")
except Exception as e:
    st.error(f"Failed to load the YOLO model weights. Please ensure 'yolov8m-football_ball_only.pt' is accessible. Error: {e}")
    st.stop()


# Constants and Colors
names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
color_ball = (0, 255, 255)
color_referee = (200, 200, 200)
color_possession = (0, 255, 0)
TEAM_A_COLOR = (0, 0, 255) # Blue (B, G, R) for darker jersey
TEAM_B_COLOR = (255, 0, 0) # Red (B, G, R) for lighter jersey


# --- Core Analysis Functions (Adapted from user's script) ---

def get_average_color(frame, box):
    """Calculates the average BGR color of a detected bounding box (ROI)."""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10: # Ensure ROI is valid
        return np.array([0.0, 0.0, 0.0])
    return np.mean(roi.reshape(-1,3), axis=0)

def assign_team(player_id, color, team_colors):
    """
    Assigns a player ID to one of two teams based on their jersey color.
    This logic needs to be robust for the app to work correctly.
    It currently attempts to group players by similar colors dynamically.
    """
    # Simple heuristic to assign two distinct teams
    if player_id not in team_colors:
        # If no teams are established, establish the first team
        if not team_colors:
            team_colors[player_id] = {'color': color, 'team_name': 'Team A', 'draw_color': TEAM_A_COLOR}
        else:
            # Try to assign to an existing team based on color distance
            min_dist = float('inf')
            assigned_team_id = None
            
            # Group existing unique team representations (if multiple players were assigned the same initial color)
            unique_teams = {}
            for pid, data in team_colors.items():
                if data['team_name'] not in unique_teams:
                    unique_teams[data['team_name']] = data['color']

            # If only one team exists, establish the second one
            if len(unique_teams) < 2:
                # Find the existing team color
                existing_color = next(iter(unique_teams.values()))
                dist = np.linalg.norm(color - existing_color)
                
                # If the color is significantly different, create Team B
                if dist > 40: # Threshold for difference
                    team_colors[player_id] = {'color': color, 'team_name': 'Team B', 'draw_color': TEAM_B_COLOR}
                else:
                    # Otherwise, assign to the existing team
                    team_data = next(iter(team_colors.values()))
                    team_colors[player_id] = team_data.copy()

            # If two teams are already established, classify the new player
            else:
                for team_name, team_avg_color in unique_teams.items():
                    dist = np.linalg.norm(color - team_avg_color)
                    if dist < min_dist:
                        min_dist = dist
                        assigned_team_id = team_name
                
                # Assign to the closest team if the distance is small enough
                if min_dist < 40:
                    data = next(data for data in team_colors.values() if data['team_name'] == assigned_team_id)
                    team_colors[player_id] = data.copy()
                else:
                    # Fallback: if color is far from both, this might be a tracking error or a third jersey color
                    # For simplicity, we just assign to the smaller team or Team A
                    team_counts = defaultdict(int)
                    for data in team_colors.values():
                        team_counts[data['team_name']] += 1
                    
                    if team_counts['Team A'] <= team_counts['Team B']:
                        team_colors[player_id] = {'color': color, 'team_name': 'Team A', 'draw_color': TEAM_A_COLOR}
                    else:
                        team_colors[player_id] = {'color': color, 'team_name': 'Team B', 'draw_color': TEAM_B_COLOR}
                        
    return team_colors[player_id]['team_name'], team_colors[player_id]['draw_color']

@st.cache_data
def run_tracking(video_path):
    """
    Runs the YOLO tracking and analysis on the video.
    Returns the path to the annotated video and the final statistics.
    """
    # Use a temporary file for the output video
    temp_output_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_output_dir, "football_tracking_output.mp4")

    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use 'mp4v' for cross-platform compatibility, though 'H264' or 'XVID' might be needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Initialize tracking data
    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors = {} # Stores assigned team info per player ID
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    st.write("Starting video processing and tracking...")
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    # Run the model stream (this part is critical for performance)
    results = model.track(
        source=video_path,
        conf=0.4,
        iou=0.5,
        tracker="botsort.yaml", # Ensure this tracker is configured/available if needed
        persist=True,
        stream=True
    )

    # Process results frame by frame
    for frame_data in results:
        frame_count += 1
        frame = frame_data.orig_img.copy()

        if frame_data.boxes.id is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        balls, players = [], []
        
        # 1. Process detections and assign teams
        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            if cls == 0:  # Ball
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ball, 2)

            elif cls in [1, 2]:  # Player or Goalkeeper
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                team_name, draw_color = assign_team(track_id, avg_color, team_colors)
                
                players.append((track_id, (x1, y1, x2, y2), team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)

            else: # Referee
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_referee, 2)


        # 2. Determine Possession
        current_owner_id = None
        current_owner_team = None
        
        if len(balls) > 0 and len(players) > 0:
            bx1, by1, bx2, by2 = balls[0][1]
            ball_center = np.array([(bx1 + bx2)/2, (by1 + by2)/2])

            min_dist = float('inf')
            
            for player_id, box, team_name in players:
                px1, py1, px2, py2 = box
                player_center = np.array([(px1 + px2)/2, (py1 + py2)/2])
                dist = np.linalg.norm(ball_center - player_center)
                
                if dist < min_dist:
                    min_dist = dist
                    current_owner_id = player_id
                    current_owner_team = team_name
            
            # If a player is close enough to the ball (possession distance)
            if min_dist < 90:
                possession_counter[current_owner_id] += 1
                team_possession_counter[current_owner_team] += 1

                # Check for a pass event
                if last_owner_id is not None and current_owner_id != last_owner_id:
                    # Only count a pass if the previous owner was also in possession (not just transition frame)
                    if possession_counter[last_owner_id] > 0: 
                        passes.append((last_owner_id, current_owner_id, team_colors[last_owner_id]['team_name'], current_owner_team))
                        team_passes_counter[current_owner_team] += 1
                        
                last_owner_id = current_owner_id
            else:
                # Ball is free, no one has possession this frame
                pass 
                
        # 3. Annotate Possession on Frame
        if current_owner_id is not None:
            for player_id, box, team_name in players:
                if player_id == current_owner_id:
                    px1, py1, px2, py2 = box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                    cv2.putText(frame, f"POSSESSION: {team_name} #{player_id}",
                                (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        # 4. Draw Stats on Frame
        start_y = 30
        
        # Display Team Possession
        offset = start_y
        for team_name, count in team_possession_counter.items():
            team_color = TEAM_A_COLOR if team_name == 'Team A' else TEAM_B_COLOR
            cv2.putText(frame, f"{team_name} Possession: {count} frames",
                        (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, team_color, 2)
            offset += 30
            
        # Display Total Passes
        cv2.putText(frame, f"Total Passes: {len(passes)}", (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ball, 2)
        offset += 30
        
        # Write frame to output video
        out.write(frame)
        
        # Update progress bar
        if total_frames > 0:
            progress = min(1.0, frame_count / total_frames)
            progress_bar.progress(progress)
            
    # Release resources
    cap.release()
    out.release()
    progress_bar.empty()
    st.success("Video analysis complete!")
    
    # Return output path and statistics
    stats = {
        'possession_player': dict(possession_counter),
        'possession_team': dict(team_possession_counter),
        'passes_list': passes,
        'passes_team': dict(team_passes_counter)
    }
    
    return output_path, stats

# --- Streamlit Application Flow ---

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Use a temporary directory to save the uploaded video file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        video_path = temp_input_file.name

    st.info("Video uploaded. Starting analysis. This may take a few moments depending on video length.")

    # Run the tracking function and get results
    try:
        output_video_path, stats = run_tracking(video_path)
    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        st.stop()
        
    # --- Display Results ---
    st.markdown("## Analyzed Video Output")
    st.video(output_video_path)

    st.markdown("---")
    st.markdown("## Tracking and Possession Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Team Possession (Frames)")
        if stats['possession_team']:
            team_data = {
                'Team': list(stats['possession_team'].keys()),
                'Frames': list(stats['possession_team'].values())
            }
            st.dataframe(team_data, use_container_width=True, hide_index=True)
        else:
            st.info("No team possession data recorded.")

        st.markdown("### Total Passes by Team")
        if stats['passes_team']:
            pass_data = {
                'Team': list(stats['passes_team'].keys()),
                'Passes': list(stats['passes_team'].values())
            }
            st.dataframe(pass_data, use_container_width=True, hide_index=True)
        else:
            st.info("No pass data recorded.")

    with col2:
        st.markdown("### Individual Player Possession (Frames)")
        if stats['possession_player']:
            player_data = {
                'Player ID': list(stats['possession_player'].keys()),
                'Frames': list(stats['possession_player'].values())
            }
            st.dataframe(player_data, use_container_width=True, hide_index=True)
        else:
            st.info("No individual player possession data recorded.")

        st.markdown("### Pass Log")
        if stats['passes_list']:
            pass_log = [{
                'From ID': f_id, 
                'To ID': t_id, 
                'From Team': f_team, 
                'To Team': t_team
            } for f_id, t_id, f_team, t_team in stats['passes_list']]
            st.dataframe(pass_log, use_container_width=True, hide_index=True)
        else:
            st.info("No passes detected.")
            
    # Clean up the temporary files (important for resource management)
    try:
        os.unlink(video_path)
        os.rmdir(temp_output_dir)
    except Exception as e:
        st.warning(f"Could not clean up temporary files: {e}")
        
    st.markdown("---")

else:
    st.info("Please upload a video to begin the analysis.")
