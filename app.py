from datetime import datetime
import threading
import time
import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (
    ClientSettings,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ==============================================================================
# Page Configuration & Modern Theme Styling
# ==============================================================================
st.set_page_config(
    page_title="Parallax AI Proctor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .status-banner {
        padding: 14px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 1.05rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .status-clean {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
        border: 1px solid #10b981;
        color: #34d399;
    }
    
    .status-warning {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.25));
        border: 1px solid #f59e0b;
        color: #fbbf24;
    }
    
    .status-cheat {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.2), rgba(185, 28, 28, 0.35));
        border: 1px solid #ef4444;
        color: #f87171;
    }
    
    .card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Proctoring Math & Logic Engine
# ==============================================================================
def update_cheat_percentage(current, previous):
    """Calculates cheat probability matching detection.py formula"""
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current


def compute_frame_weight(global_cheat, x_cheat, y_cheat):
    """Calculates increment weight based on gaze state"""
    if global_cheat == 0:
        if x_cheat == 0:
            return 0.2 if y_cheat != 0 else 0.0
        else:
            return 0.15 if y_cheat != 0 else 0.10
    else:
        if x_cheat == 0:
            return 0.55 if y_cheat != 0 else 0.0
        else:
            return 0.50 if y_cheat != 0 else 0.60


# ==============================================================================
# WebRTC Video Processor (Runs directly on browser video stream)
# ==============================================================================
class ProctoringVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Dynamic telemetry variables
        self.gaze_text = "Waiting for Camera..."
        self.yaw_angle = 0.0
        self.pitch_angle = 0.0
        self.percentage_cheat = 0.0
        self.global_cheat = 0
        self.total_violations = 0
        self.last_log_time = 0
        self.incidents = []

        # Configurable thresholds
        self.yaw_threshold = 10
        self.pitch_threshold = 8
        self.cheat_threshold = 0.6
        self.draw_mesh = True
        self.draw_vector = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Mirror horizontally for natural selfie view
        image = cv2.flip(image, 1)
        img_h, img_w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        gaze_text = "Forward"
        gaze_color = (0, 255, 0)
        x_axis_cheat = 0
        y_axis_cheat = 0
        pitch = 0.0
        yaw = 0.0

        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if self.draw_mesh:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None
                    )

                face_2d = []
                face_3d = []
                nose_2d = (0, 0)
                nose_3d = (0, 0, 0)

                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1.0 * img_w
                cam_matrix = np.array([
                    [focal_length, 0, img_h / 2],
                    [0, focal_length, img_w / 2],
                    [0, 0, 1]
                ])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                if success:
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    pitch = angles[0] * 360
                    yaw = angles[1] * 360

                    if yaw < -self.yaw_threshold:
                        gaze_text = "Looking Left"
                        gaze_color = (0, 69, 255)
                        x_axis_cheat = 1
                    elif yaw > self.yaw_threshold:
                        gaze_text = "Looking Right"
                        gaze_color = (0, 69, 255)
                        x_axis_cheat = 1
                    elif pitch < -self.pitch_threshold:
                        gaze_text = "Looking Down"
                        gaze_color = (0, 140, 255)
                        y_axis_cheat = 1
                    else:
                        gaze_text = "Forward"
                        gaze_color = (0, 255, 0)

                    if self.draw_vector:
                        nose_3d_proj, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_3d_proj[0][0][0]), int(nose_3d_proj[0][0][1]))
                        cv2.line(image, p1, p2, (255, 255, 0), 2)
        else:
            gaze_text = "No Face Detected"
            gaze_color = (0, 0, 255)
            x_axis_cheat = 1
            y_axis_cheat = 1

        # Calculate cheat probability
        with self.lock:
            weight = compute_frame_weight(self.global_cheat, x_axis_cheat, y_axis_cheat)
            self.percentage_cheat = update_cheat_percentage(weight, self.percentage_cheat)

            if self.percentage_cheat > self.cheat_threshold:
                self.global_cheat = 1
            else:
                self.global_cheat = 0

            self.gaze_text = gaze_text
            self.yaw_angle = yaw
            self.pitch_angle = pitch

            # Log violation events
            now = time.time()
            if (x_axis_cheat or y_axis_cheat or self.global_cheat) and (now - self.last_log_time > 2.0):
                self.incidents.insert(0, {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Violation": gaze_text,
                    "Cheat Score": f"{self.percentage_cheat:.2f}"
                })
                self.total_violations += 1
                self.last_log_time = now

        # On-screen HUD elements
        cv2.putText(image, f"Status: {gaze_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, gaze_color, 2)
        cv2.putText(image, f"Cheat Score: {self.percentage_cheat:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if self.global_cheat:
            cv2.putText(image, "CHEATING DETECTED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ==============================================================================
# UI & Dashboard Layout
# ==============================================================================
st.title("🛡️ Parallax AI Proctoring Dashboard")
st.caption("Real-Time Autonomous Exam Integrity & Head-Pose Monitoring (Cloud & Browser WebRTC Supported)")

# Sidebar Settings
st.sidebar.header("🎯 Sensitivity Settings")
yaw_threshold = st.sidebar.slider("Gaze Yaw Threshold (Left/Right °)", 5, 30, 10)
pitch_threshold = st.sidebar.slider("Gaze Pitch Threshold (Down °)", 5, 25, 8)
cheat_threshold = st.sidebar.slider("Cheating Threshold", 0.3, 0.9, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Visual Overlays")
draw_mesh = st.sidebar.checkbox("Show Face Mesh Contours", value=True)
draw_vector = st.sidebar.checkbox("Show 3D Gaze Vector", value=True)

# Google public STUN servers for WebRTC NAT traversal across networks
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Status Banner Placeholder
status_container = st.empty()

# Metric Columns
m1, m2, m3, m4 = st.columns(4)
with m1:
    prob_metric = st.empty()
with m2:
    gaze_metric = st.empty()
with m3:
    yaw_metric = st.empty()
with m4:
    events_metric = st.empty()

# Workspace Columns
col_video, col_insights = st.columns([1.3, 1.0])

with col_video:
    st.subheader("📹 Candidate Camera Feed (Browser WebRTC)")
    st.info("Click **'START'** below to grant browser webcam access.")
    webrtc_ctx = webrtc_streamer(
        key="parallax-proctor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=ProctoringVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_insights:
    st.subheader("📋 Session Incident Log")
    log_container = st.empty()

# Connect sidebar sliders to live processor
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.yaw_threshold = yaw_threshold
    webrtc_ctx.video_processor.pitch_threshold = pitch_threshold
    webrtc_ctx.video_processor.cheat_threshold = cheat_threshold
    webrtc_ctx.video_processor.draw_mesh = draw_mesh
    webrtc_ctx.video_processor.draw_vector = draw_vector

    score = webrtc_ctx.video_processor.percentage_cheat
    gaze = webrtc_ctx.video_processor.gaze_text
    yaw = webrtc_ctx.video_processor.yaw_angle
    violations = webrtc_ctx.video_processor.total_violations
    incidents = webrtc_ctx.video_processor.incidents

    # Update Status Banner
    if webrtc_ctx.video_processor.global_cheat:
        status_container.markdown(f"""
            <div class="status-banner status-cheat">
                <span>🚨 ALERT: SUSPICIOUS BEHAVIOR DETECTED (Score: {score:.2f})</span>
                <span>CHEATING FLAGGED</span>
            </div>
        """, unsafe_allow_html=True)
    elif score > 0.35:
        status_container.markdown(f"""
            <div class="status-banner status-warning">
                <span>⚠️ CAUTION: Irregular gaze activity detected</span>
                <span>MONITORING CLOSELY</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        status_container.markdown("""
            <div class="status-banner status-clean">
                <span>🟢 CANDIDATE NORMAL — Active proctoring stream</span>
                <span>INTEGRITY VERIFIED</span>
            </div>
        """, unsafe_allow_html=True)

    # Update Metrics
    prob_metric.metric("Cheat Probability", f"{score * 100:.1f}%")
    gaze_metric.metric("Gaze Direction", gaze)
    yaw_metric.metric("Yaw / Pitch Angle", f"{yaw:.1f}°")
    events_metric.metric("Flagged Events", f"{violations}")

    # Incident Log
    if incidents:
        log_container.dataframe(pd.DataFrame(incidents[:10]), use_container_width=True)
    else:
        log_container.info("No suspicious incidents detected yet.")

else:
    status_container.markdown("""
        <div class="status-banner status-clean">
            <span>⚪ SYSTEM IDLE — Click 'START' in the camera player below to grant camera access</span>
            <span>STANDBY</span>
        </div>
    """, unsafe_allow_html=True)
    prob_metric.metric("Cheat Probability", "0.0%")
    gaze_metric.metric("Gaze Direction", "Offline")
    yaw_metric.metric("Yaw / Pitch Angle", "0.0°")
    events_metric.metric("Flagged Events", "0")
    log_container.info("Camera is offline. Start the stream to view live audit incidents.")
