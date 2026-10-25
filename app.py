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
    AudioProcessorBase,
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
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Shared State for Multi-modal Telemetry (Video + Audio)
# ==============================================================================
class ProctorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.audio_amplitude = 0.0
        self.audio_cheat = 0
        self.audio_threshold = 15.0


@st.cache_resource
def get_shared_state():
    return ProctorState()


shared_state = get_shared_state()


# ==============================================================================
# Proctoring Math & Detection Logic
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


def compute_frame_weight(global_cheat, x_cheat, y_cheat, audio_cheat):
    """Calculates increment weight combining Gaze + Audio events"""
    if global_cheat == 0:
        if x_cheat == 0:
            if y_cheat == 0:
                return 0.25 if audio_cheat else 0.0
            else:
                return 0.40 if audio_cheat else 0.20
        else:
            if y_cheat == 0:
                return 0.40 if audio_cheat else 0.10
            else:
                return 0.25 if audio_cheat else 0.15
    else:
        if x_cheat == 0:
            if y_cheat == 0:
                return 0.55 if audio_cheat else 0.0
            else:
                return 0.85 if audio_cheat else 0.55
        else:
            if y_cheat == 0:
                return 0.85 if audio_cheat else 0.60
            else:
                return 0.85 if audio_cheat else 0.50


# ==============================================================================
# WebRTC Audio Processor (Analyzes microphone sound level in real-time)
# ==============================================================================
class ProctoringAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.state = shared_state

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            samples = frame.to_ndarray()
            if np.issubdtype(samples.dtype, np.integer):
                samples = samples.astype(np.float64) / 32768.0
            else:
                samples = samples.astype(np.float64)

            rms = float(np.sqrt(np.mean(samples**2)) * 1000.0)
            with self.state.lock:
                self.state.audio_amplitude = rms
                if rms > self.state.audio_threshold:
                    self.state.audio_cheat = 1
                else:
                    self.state.audio_cheat = 0
        except Exception:
            pass

        return frame


# ==============================================================================
# WebRTC Video Processor (Face Mesh + Gaze Estimation + Audio Fusion)
# ==============================================================================
class ProctoringVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.state = shared_state
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Live telemetry
        self.gaze_text = "Forward"
        self.yaw_angle = 0.0
        self.pitch_angle = 0.0
        self.percentage_cheat = 0.0
        self.global_cheat = 0
        self.total_violations = 0
        self.last_log_time = 0
        self.incidents = []
        self.prob_history = [0.0] * 60

        # Sliders / Configuration
        self.yaw_threshold = 10
        self.pitch_threshold = 8
        self.cheat_threshold = 0.60
        self.draw_mesh = True
        self.draw_vector = True

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Mirror for natural selfie perspective
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

        # Read shared audio state
        with self.state.lock:
            audio_cheat = self.state.audio_cheat
            audio_amp = self.state.audio_amplitude

        with self.lock:
            weight = compute_frame_weight(self.global_cheat, x_axis_cheat, y_axis_cheat, audio_cheat)
            self.percentage_cheat = update_cheat_percentage(weight, self.percentage_cheat)

            if self.percentage_cheat > self.cheat_threshold:
                self.global_cheat = 1
            else:
                self.global_cheat = 0

            self.gaze_text = gaze_text
            self.yaw_angle = yaw
            self.pitch_angle = pitch

            # Rolling chart history update
            self.prob_history.pop(0)
            self.prob_history.append(float(self.percentage_cheat))

            # Incident logging
            now = time.time()
            if (x_axis_cheat or y_axis_cheat or audio_cheat or self.global_cheat) and (now - self.last_log_time > 2.0):
                reasons = []
                if x_axis_cheat or y_axis_cheat:
                    reasons.append(gaze_text)
                if audio_cheat:
                    reasons.append(f"Noise (RMS: {audio_amp:.1f})")
                if self.global_cheat:
                    reasons.append("High Suspicion")

                self.incidents.insert(0, {
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Incident": " + ".join(reasons),
                    "Cheat Score": f"{self.percentage_cheat:.2f}"
                })
                self.total_violations += 1
                self.last_log_time = now

        # On-frame HUD status overlay
        cv2.putText(image, f"Status: {gaze_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, gaze_color, 2)
        cv2.putText(image, f"Cheat Score: {self.percentage_cheat:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if audio_cheat:
            cv2.putText(image, f"AUDIO ALERT: {audio_amp:.1f} RMS", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        if self.global_cheat:
            cv2.putText(image, "CHEATING DETECTED", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ==============================================================================
# UI Dashboard Layout & Controls
# ==============================================================================
st.title("🛡️ Parallax AI Proctoring Dashboard")
st.caption("Real-Time Autonomous Exam Integrity & Head-Pose / Audio Monitoring")

# Sidebar
st.sidebar.header("⚙️ Session Controls")
st.sidebar.info("Audio & Video streaming enabled via browser WebRTC.")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Sensitivity Settings")
yaw_threshold = st.sidebar.slider("Gaze Yaw Threshold (Left/Right °)", 5, 30, 10)
pitch_threshold = st.sidebar.slider("Gaze Pitch Threshold (Down °)", 5, 25, 8)
audio_threshold = st.sidebar.slider("Audio Amplitude Limit (RMS)", 1, 50, 15)
cheat_threshold = st.sidebar.slider("Cheating Threshold", 0.3, 0.9, 0.6, 0.05)

# Sync audio limit immediately to shared state
with shared_state.lock:
    shared_state.audio_threshold = float(audio_threshold)

st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Visual Overlays")
draw_mesh = st.sidebar.checkbox("Show Face Mesh Contours", value=True)
draw_vector = st.sidebar.checkbox("Show 3D Gaze Vector", value=True)

# Google public STUN server for WebRTC NAT traversal
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Status Banner Placeholder
status_placeholder = st.empty()

# 4 Metric Cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    prob_metric = st.empty()
with m2:
    gaze_metric = st.empty()
with m3:
    audio_metric = st.empty()
with m4:
    events_metric = st.empty()

# Main Workspace Columns
col_video, col_charts = st.columns([1.3, 1.0])

with col_video:
    st.subheader("📹 Candidate Video & Audio Stream")
    webrtc_ctx = webrtc_streamer(
        key="parallax-proctor",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=ProctoringVideoProcessor,
        audio_processor_factory=ProctoringAudioProcessor,
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

with col_charts:
    st.subheader("📈 Real-Time Cheat Probability Trend")
    chart_placeholder = st.empty()
    st.subheader("📋 Violation Incident Log")
    log_placeholder = st.empty()

# Reactive Real-Time Polling Loop while stream is active
if webrtc_ctx.state.playing:
    while webrtc_ctx.state.playing:
        if webrtc_ctx.video_processor:
            # Sync video processor parameters
            webrtc_ctx.video_processor.yaw_threshold = yaw_threshold
            webrtc_ctx.video_processor.pitch_threshold = pitch_threshold
            webrtc_ctx.video_processor.cheat_threshold = cheat_threshold
            webrtc_ctx.video_processor.draw_mesh = draw_mesh
            webrtc_ctx.video_processor.draw_vector = draw_vector

            with webrtc_ctx.video_processor.lock:
                score = webrtc_ctx.video_processor.percentage_cheat
                gaze = webrtc_ctx.video_processor.gaze_text
                yaw = webrtc_ctx.video_processor.yaw_angle
                violations = webrtc_ctx.video_processor.total_violations
                history = list(webrtc_ctx.video_processor.prob_history)
                incidents = list(webrtc_ctx.video_processor.incidents)
                is_cheat = webrtc_ctx.video_processor.global_cheat

            with shared_state.lock:
                audio_amp = shared_state.audio_amplitude
                audio_flag = shared_state.audio_cheat

            # Status Banner
            if is_cheat:
                status_placeholder.markdown(f"""
                    <div class="status-banner status-cheat">
                        <span>🚨 ALERT: SUSPICIOUS BEHAVIOR DETECTED (Score: {score:.2f})</span>
                        <span>CHEATING FLAGGED</span>
                    </div>
                """, unsafe_allow_html=True)
            elif score > 0.35 or audio_flag:
                status_placeholder.markdown(f"""
                    <div class="status-banner status-warning">
                        <span>⚠️ CAUTION: Irregular gaze or sound activity detected</span>
                        <span>MONITORING CLOSELY</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                status_placeholder.markdown("""
                    <div class="status-banner status-clean">
                        <span>🟢 CANDIDATE NORMAL — Active proctoring stream</span>
                        <span>INTEGRITY VERIFIED</span>
                    </div>
                """, unsafe_allow_html=True)

            # 4 Metric Cards
            prob_metric.metric("Cheat Probability", f"{score * 100:.1f}%")
            gaze_metric.metric("Gaze Direction", gaze)
            audio_metric.metric("Noise RMS", f"{audio_amp:.1f}", delta="Spike" if audio_flag else "Normal")
            events_metric.metric("Flagged Events", f"{violations}")

            # Live Rolling Trend Chart
            chart_placeholder.line_chart(pd.DataFrame({"Cheat Probability": history}), height=240)

            # Incident Log Table
            if incidents:
                log_placeholder.dataframe(pd.DataFrame(incidents[:10]), height=200)

        time.sleep(0.2)

    # Export Report in Sidebar
    if incidents:
        csv_data = pd.DataFrame(incidents).to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "📥 Download Audit Report",
            data=csv_data,
            file_name=f"proctor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    # Idle Standby State
    status_placeholder.markdown("""
        <div class="status-banner status-clean">
            <span>⚪ SYSTEM IDLE — Click 'START' below to grant camera and microphone access</span>
            <span>STANDBY</span>
        </div>
    """, unsafe_allow_html=True)
    prob_metric.metric("Cheat Probability", "0.0%")
    gaze_metric.metric("Gaze Direction", "Offline")
    audio_metric.metric("Noise RMS", "0.0")
    events_metric.metric("Flagged Events", "0")
    chart_placeholder.line_chart(pd.DataFrame({"Cheat Probability": [0.0] * 50}), height=240)
    log_placeholder.info("Camera is offline. Start the stream to view live audit incidents.")
