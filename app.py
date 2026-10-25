import time
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import sounddevice as sd
import streamlit as st

# ==============================================================================
# Page Configuration & Custom CSS
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
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Proctoring Engine Logic
# ==============================================================================
class AudioMonitor:
    def __init__(self, threshold=20):
        self.threshold = threshold
        self.amplitude = 0.0
        self.is_cheat = 0
        self.stream = None
        self._running = False
        
    def _audio_callback(self, indata, frames, time_info, status):
        try:
            rms = float(np.sqrt(np.mean(indata**2)) * 1000)
            self.amplitude = rms
            if rms > self.threshold:
                self.is_cheat = 1
            else:
                self.is_cheat = 0
        except Exception:
            pass

    def start(self):
        try:
            self.stream = sd.InputStream(callback=self._audio_callback, channels=1, samplerate=44100)
            self.stream.start()
            self._running = True
        except Exception as e:
            st.sidebar.warning(f"Microphone access notice: {e}")
            self._running = False

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self._running = False


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


def compute_frame_cheat_weight(global_cheat, x_cheat, y_cheat, audio_cheat):
    """Calculates increment weight based on behavioral state"""
    if global_cheat == 0:
        if x_cheat == 0:
            if y_cheat == 0:
                return 0.2 if audio_cheat else 0.0
            else:
                return 0.4 if audio_cheat else 0.2
        else:
            if y_cheat == 0:
                return 0.4 if audio_cheat else 0.1
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
                return 0.85 if audio_cheat else 0.6
            else:
                return 0.85 if audio_cheat else 0.5


# ==============================================================================
# UI & Dashboard Layout
# ==============================================================================
st.title("🛡️ Parallax AI Proctoring Dashboard")
st.caption("Real-Time Autonomous Exam Integrity & Head-Pose / Audio Monitoring")

# Sidebar Configuration
st.sidebar.header("⚙️ Session Controls")
session_active = st.sidebar.toggle("Start Proctoring Session", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Sensitivity Settings")
yaw_threshold = st.sidebar.slider("Gaze Yaw Threshold (Left/Right °)", 5, 30, 10)
pitch_threshold = st.sidebar.slider("Gaze Pitch Threshold (Down °)", 5, 25, 8)
audio_threshold = st.sidebar.slider("Audio Amplitude Limit", 5, 50, 20)
cheat_threshold = st.sidebar.slider("Cheating Threshold", 0.3, 0.9, 0.6, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Visual Overlays")
draw_mesh = st.sidebar.checkbox("Show Face Mesh Contours", value=True)
draw_vector = st.sidebar.checkbox("Show 3D Gaze Vector", value=True)

# Main UI Placeholders
status_placeholder = st.empty()

# Metric Columns
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
    st.subheader("📹 Candidate Video Stream")
    video_placeholder = st.empty()

with col_charts:
    st.subheader("📈 Real-Time Cheat Probability Trend")
    chart_placeholder = st.empty()
    st.subheader("📋 Violation Incident Log")
    log_placeholder = st.empty()

# Initial idle state
if not session_active:
    status_placeholder.markdown("""
        <div class="status-banner status-clean">
            <span>⚪ SYSTEM IDLE — Toggle 'Start Proctoring Session' in the sidebar to begin monitoring</span>
            <span>STANDBY</span>
        </div>
    """, unsafe_allow_html=True)
    video_placeholder.info("Camera is offline. Toggle 'Start Proctoring Session' in the sidebar to start live examination.")
    chart_placeholder.line_chart(pd.DataFrame({"Cheat Probability": [0] * 50}))
    log_placeholder.dataframe(pd.DataFrame(columns=["Timestamp", "Incident", "Value"]), use_container_width=True)

# Active Session Execution Loop
else:
    # Initialize Audio Monitor
    audio_monitor = AudioMonitor(threshold=audio_threshold)
    audio_monitor.start()

    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Open Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not access the webcam. Please verify camera permissions.")
    else:
        # History buffers for charts and logs
        prob_history = [0.0] * 60
        incidents = []
        percentage_cheat = 0.0
        global_cheat = 0
        total_violations = 0
        last_log_time = 0

        try:
            while session_active:
                success, image = cap.read()
                if not success:
                    st.warning("Failed to receive video frame.")
                    break

                # Flip horizontally for selfie view and convert colors
                image = cv2.flip(image, 1)
                img_h, img_w, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                results = face_mesh.process(image_rgb)
                image_rgb.flags.writeable = True

                # Default values
                gaze_text = "Forward"
                gaze_color = (0, 255, 0)
                x_axis_cheat = 0
                y_axis_cheat = 0
                yaw_angle = 0.0
                pitch_angle = 0.0

                face_ids = [33, 263, 1, 61, 291, 199]
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        if draw_mesh:
                            mp_drawing.draw_landmarks(
                                image=image_rgb,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
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

                        success_pnp, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                        if success_pnp:
                            rmat, _ = cv2.Rodrigues(rot_vec)
                            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                            pitch_angle = angles[0] * 360
                            yaw_angle = angles[1] * 360

                            # Gaze classification
                            if yaw_angle < -yaw_threshold:
                                gaze_text = "Looking Left"
                                gaze_color = (255, 69, 0)
                                x_axis_cheat = 1
                            elif yaw_angle > yaw_threshold:
                                gaze_text = "Looking Right"
                                gaze_color = (255, 69, 0)
                                x_axis_cheat = 1
                            elif pitch_angle < -pitch_threshold:
                                gaze_text = "Looking Down"
                                gaze_color = (255, 140, 0)
                                y_axis_cheat = 1
                            else:
                                gaze_text = "Forward"
                                gaze_color = (0, 255, 0)

                            if draw_vector:
                                nose_3d_proj, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                                p2 = (int(nose_3d_proj[0][0][0]), int(nose_3d_proj[0][0][1]))
                                cv2.line(image_rgb, p1, p2, (255, 255, 0), 2)

                else:
                    gaze_text = "No Face Detected"
                    gaze_color = (255, 0, 0)
                    x_axis_cheat = 1
                    y_axis_cheat = 1

                # Audio measurement
                audio_amp = audio_monitor.amplitude
                audio_cheat = audio_monitor.is_cheat

                # Compute Cheat Probability
                weight = compute_frame_cheat_weight(global_cheat, x_axis_cheat, y_axis_cheat, audio_cheat)
                percentage_cheat = update_cheat_percentage(weight, percentage_cheat)

                if percentage_cheat > cheat_threshold:
                    global_cheat = 1
                else:
                    global_cheat = 0

                # Buffer history for rolling line chart
                prob_history.pop(0)
                prob_history.append(float(percentage_cheat))

                # Log incidents with rate limiting (max 1 log every 2 seconds)
                current_time = time.time()
                if (x_axis_cheat or y_axis_cheat or audio_cheat or global_cheat) and (current_time - last_log_time > 2.0):
                    reason = []
                    if x_axis_cheat or y_axis_cheat:
                        reason.append(gaze_text)
                    if audio_cheat:
                        reason.append(f"Noise (RMS: {audio_amp:.1f})")
                    if global_cheat:
                        reason.append("Threshold Exceeded")

                    incidents.insert(0, {
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Incident": " + ".join(reason),
                        "Value": f"Cheat Score: {percentage_cheat:.2f}"
                    })
                    total_violations += 1
                    last_log_time = current_time

                # Draw overlay tags on frame
                cv2.putText(image_rgb, f"Status: {gaze_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, gaze_color, 2)
                cv2.putText(image_rgb, f"Cheat Score: {percentage_cheat:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if audio_cheat:
                    cv2.putText(image_rgb, "AUDIO ALERT", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2)

                # Render Video Frame
                video_placeholder.image(image_rgb, channels="RGB", use_container_width=True)

                # Render Top Status Banner
                if global_cheat:
                    status_placeholder.markdown(f"""
                        <div class="status-banner status-cheat">
                            <span>🚨 ALERT: SUSPICIOUS BEHAVIOR DETECTED (Score: {percentage_cheat:.2f})</span>
                            <span>CHEATING FLAGGED</span>
                        </div>
                    """, unsafe_allow_html=True)
                elif percentage_cheat > 0.35 or audio_cheat or x_axis_cheat:
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

                # Render Metric Cards
                prob_metric.metric("Cheat Probability", f"{percentage_cheat * 100:.1f}%", delta=f"{weight:.2f}")
                gaze_metric.metric("Gaze Direction", gaze_text, delta=f"Yaw: {yaw_angle:.1f}°")
                audio_metric.metric("Noise RMS", f"{audio_amp:.1f}", delta="Alert" if audio_cheat else "Normal")
                events_metric.metric("Flagged Events", f"{total_violations}")

                # Render Live Chart
                chart_placeholder.line_chart(
                    pd.DataFrame({"Cheat Probability": prob_history}),
                    height=240
                )

                # Render Incident Table
                if incidents:
                    log_placeholder.dataframe(
                        pd.DataFrame(incidents[:10]),
                        use_container_width=True,
                        height=200
                    )

                time.sleep(0.04)

        finally:
            cap.release()
            audio_monitor.stop()

# Export Log in Sidebar if any events recorded
if session_active and 'incidents' in locals() and incidents:
    csv_data = pd.DataFrame(incidents).to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "📥 Download Audit Report",
        data=csv_data,
        file_name=f"proctor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
