import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import json
import os
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sounddevice as sd
import threading


#warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Import your existing modules
import audio
import detection
import head_pose

# Configure Streamlit with enhanced settings
st.set_page_config(
    page_title="Parallax AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for live graph data
if 'violation_history' not in st.session_state:
    st.session_state.violation_history = []
if 'risk_history' not in st.session_state:
    st.session_state.risk_history = []
if 'time_history' not in st.session_state:
    st.session_state.time_history = []
if 'risk_hist' not in st.session_state:
    st.session_state.risk_hist = []

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Custom Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f2f6;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    /* Status Cards */
    .status-normal {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        animation: pulse 2s infinite;
    }
    
    .status-violation {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        border: none;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Metrics Enhancement */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Progress Bars */
    .progress-bar {
        background: #f0f2f6;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .progress-normal { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); }
    .progress-warning { background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); }
    .progress-danger { background: linear-gradient(90deg, #ff6b6b 0%, #ee5a52 100%); }
    
    /* Alert Boxes */
    .custom-alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
        animation: slideIn 0.3s ease;
    }
    
    .alert-success {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .alert-warning {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .alert-danger {
        background: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Video Container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        background: white;
        padding: 1rem;
    }
    
    /* Live Indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        background: #ff4444;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 1rem;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: white;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    
    /* Charts Container */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .metric-card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Shared state file for real-time communication
STATE_FILE = "proctoring_state.json"
VIOLATIONS_LOG_FILE = "violations_log.json"

def save_state(state):
    """Save current state to file"""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except:
        pass

def load_state():
    """Load current state from file"""
    default_state = {
        'violation': False,
        'violation_type': 'Normal',
        'x_angle': 0,
        'y_angle': 0,
        'risk': 0,
        'frame_count': 0,
        'timestamp': datetime.now().isoformat(),
        'session_start': datetime.now().isoformat(),
        'total_violations': 0,
        'audio_level': 0,
        'monitoring_active': False  # Track if monitoring is actually active
    }
    
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                loaded_state = json.load(f)
                # Merge with default state to ensure all keys exist
                default_state.update(loaded_state)
                return default_state
    except:
        pass
    return default_state

def clear_violation_logs():
    """Clear existing violation logs"""
    try:
        if os.path.exists(VIOLATIONS_LOG_FILE):
            os.remove(VIOLATIONS_LOG_FILE)
    except:
        pass

def save_violation_log(violation_data):
    """Save violation to log file - only for actual violations"""
    # Don't log if it's a normal state or default values
    if violation_data.get('type') == 'Normal' or violation_data.get('risk', 0) == 0:
        return
        
    try:
        violations = []
        if os.path.exists(VIOLATIONS_LOG_FILE):
            with open(VIOLATIONS_LOG_FILE, 'r') as f:
                violations = json.load(f)
        
        violations.append(violation_data)
        
        # Keep only last 100 violations
        if len(violations) > 100:
            violations = violations[-100:]
            
        with open(VIOLATIONS_LOG_FILE, 'w') as f:
            json.dump(violations, f)
    except:
        pass

def load_violations_log():
    """Load violations log"""
    try:
        if os.path.exists(VIOLATIONS_LOG_FILE):
            with open(VIOLATIONS_LOG_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return []

class EnhancedProctoringProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,  # Reduced for better detection
                min_tracking_confidence=0.5,   # Reduced for better detection
                max_num_faces=1,
                refine_landmarks=True
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mediapipe_working = True
        except Exception as e:
            self.mediapipe_working = False
            print(f"MediaPipe failed: {e}")
        
        # Initialize audio detection (like run.py)
        self.audio_thread = None
        self.audio_running = False
        self.start_audio_detection()
        
        # Initialize detection loop (like run.py)
        self.detection_thread = None
        self.detection_running = False
        self.start_detection_loop()
        
        self.face_ids = [33, 263, 1, 61, 291, 199]
        self.frame_count = 0
        self.violation_count = 0
        self.session_start = datetime.now()
    
    def start_audio_detection(self):
        """Start audio detection thread like run.py"""
        try:
            if not self.audio_running:
                self.audio_running = True
                self.audio_thread = threading.Thread(target=self.run_audio_detection, daemon=True)
                self.audio_thread.start()
                print("Audio detection started")
        except Exception as e:
            print(f"Audio detection failed to start: {e}")
    
    def start_detection_loop(self):
        """Start detection loop thread like run.py"""
        try:
            if not self.detection_running:
                self.detection_running = True
                self.detection_thread = threading.Thread(target=self.run_detection_loop, daemon=True)
                self.detection_thread.start()
                print("Detection loop started like run.py")
        except Exception as e:
            print(f"Detection loop failed to start: {e}")
    
    def run_detection_loop(self):
        """Run detection loop that updates YDATA like detection.run_detection()"""
        try:
            # Initialize detection arrays like detection.py
            if not hasattr(detection, 'YDATA'):
                detection.YDATA = [0] * 200
            if not hasattr(detection, 'XDATA'):
                detection.XDATA = list(range(200))
            
            while self.detection_running:
                # Update YDATA with current PERCENTAGE_CHEAT (like detection.run_detection())
                detection.YDATA.pop(0)
                detection.YDATA.append(detection.PERCENTAGE_CHEAT)
                
                # Sleep same as detection.py: time.sleep(1/5)
                time.sleep(1/5)
                
        except Exception as e:
            print(f"Detection loop error: {e}")
    
    def run_audio_detection(self):
        """Run existing audio.sound() function in background like run.py"""
        try:
            # Use your existing audio.sound() function directly
            audio.sound()
        except Exception as e:
            print(f"Audio detection error: {e}")
            audio.AUDIO_CHEAT = 0
        
    def recv(self, frame):
        try:
            # Get frame
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)
            self.frame_count += 1
            
            # Initialize state
            state = {
                'violation': False,
                'violation_type': 'Normal',
                'x_angle': 0,
                'y_angle': 0,
                'risk': 0,
                'frame_count': self.frame_count,
                'timestamp': datetime.now().isoformat(),
                'session_start': self.session_start.isoformat(),
                'total_violations': self.violation_count,
                'audio_level': getattr(audio, 'SOUND_AMPLITUDE', 0),
                'monitoring_active': True
            }
            
            # Enhanced frame overlay
            self.add_enhanced_overlay(image)
            
            if self.mediapipe_working:
                # MediaPipe processing
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                results = self.face_mesh.process(rgb_image)
                rgb_image.flags.writeable = True
                image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                
                if results.multi_face_landmarks:
                    # Draw enhanced face mesh
                    for face_landmarks in results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                            None, self.mp_drawing.DrawingSpec(color=(0, 255, 100), thickness=1, circle_radius=1)
                        )
                    
                    # Process pose
                    violation_info = self.process_face_landmarks(image, results.multi_face_landmarks[0])
                    if violation_info:
                        state.update(violation_info)
                        # Update live monitoring data with cheat probability (like run.py)
                        cheat_prob = violation_info.get('cheat_probability', violation_info['risk'])
                        update_monitoring_data(cheat_prob)
                        
                        if violation_info['violation']:
                            self.violation_count += 1
                            state['total_violations'] = self.violation_count
                            # Only log actual violations (not normal states)
                            save_violation_log({
                                'timestamp': datetime.now().isoformat(),
                                'type': violation_info['violation_type'],
                                'risk': cheat_prob,  # Use cheat probability instead
                                'x_angle': violation_info['x_angle'],
                                'y_angle': violation_info['y_angle']
                            })
                    else:
                        # Update monitoring data even when no violation
                        update_monitoring_data(0.0)
                else:
                    # No face detected - enhanced guidance message
                    h, w = image.shape[:2]
                    
                    # Add colored background for visibility
                    cv2.rectangle(image, (0, h//2-60), (w, h//2+60), (0, 0, 100), -1)
                    
                    # Multi-line instruction
                    cv2.putText(image, "No Face Detected", (w//2-150, h//2-20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(image, "Please position yourself in front of camera", (w//2-250, h//2+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(image, "Ensure good lighting and face visibility", (w//2-240, h//2+35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Update monitoring with face detection failure
                    update_monitoring_data(0.8)
                    
                    state.update({
                        'violation': True,
                        'violation_type': 'No Face Detected',
                        'risk': 0.8
                    })
            
            # Save state for dashboard
            save_state(state)
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")
            
        except Exception as e:
            error_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_image, f"Error: {str(e)[:50]}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(error_image, format="bgr24")
    
    def add_enhanced_overlay(self, image):
        """Add clean overlay to video frame"""
        h, w = image.shape[:2]
        
        # Simple top banner
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Frame counter
        cv2.putText(image, f"Frame: {self.frame_count}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        
        # Status indicator
        status_text = "AI Monitoring" if self.mediapipe_working else "Fallback Mode"
        status_color = (0, 255, 100) if self.mediapipe_working else (0, 165, 255)
        cv2.putText(image, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
    def add_violation_overlay(self, image, text, color):
        """Add simple violation text overlay (deprecated - keeping for compatibility)"""
        # Just add simple text without dramatic overlay
        cv2.putText(image, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
    def process_face_landmarks(self, image, face_landmarks):
        """Process face landmarks and return violation info - CRITICAL: Updates head_pose module variables"""
        try:
            img_h, img_w = image.shape[:2]
            face_3d = []
            face_2d = []
            
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in self.face_ids:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            if len(face_2d) >= 6:
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                
                focal_length = img_w
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
                    
                    x = angles[0] * 360
                    y = angles[1] * 360
                    
                    # *** USE EXISTING HEAD_POSE.PY LOGIC ***
                    # Update head_pose module variables using existing logic from head_pose.py
                    head_pose.x = x
                    head_pose.y = y
                    
                    # Use exact logic from head_pose.py for cheat flags
                    if y < -10 or y > 10:
                        head_pose.X_AXIS_CHEAT = 1
                    else:
                        head_pose.X_AXIS_CHEAT = 0
                    
                    if x < -5:
                        head_pose.Y_AXIS_CHEAT = 1
                    else:
                        head_pose.Y_AXIS_CHEAT = 0
                    
                    # *** USE EXISTING DETECTION.PY PROCESS() FUNCTION ***
                    # Call your existing detection.process() function
                    detection.process()
                    
                    # Get results from detection.py variables
                    cheat_probability = detection.PERCENTAGE_CHEAT
                    global_cheat = detection.GLOBAL_CHEAT
                    
                    # Use cheat probability as the risk value for consistency with run.py
                    risk = cheat_probability
                    
                    # Determine violation based on global cheat flag (like detection.py)
                    if global_cheat == 1:
                        violation = True
                        if y < -10:
                            violation_type = "Looking Left"
                        elif y > 10:
                            violation_type = "Looking Right"
                        elif x < -5:
                            violation_type = "Looking Down"
                        else:
                            violation_type = "Cheating Detected"
                    else:
                        violation = False
                        violation_type = "Normal"
                    
                    # Display current values like run.py with clear visibility
                    cv2.putText(image, f"Head: X={x:.1f}° Y={y:.1f}°", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Flags: X={head_pose.X_AXIS_CHEAT} Y={head_pose.Y_AXIS_CHEAT} A={audio.AUDIO_CHEAT}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(image, f"Cheat%: {cheat_probability:.3f}", (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(image, f"Audio: {audio.SOUND_AMPLITUDE:.1f}", (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(image, f"Global: {global_cheat}", (20, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
                    
                    # Debug: Print to console (like run.py) - detection.py already prints
                    # No need to print here since detection.process() already prints
                    
                    return {
                        'violation': violation,
                        'violation_type': violation_type,
                        'x_angle': x,
                        'y_angle': y,
                        'risk': risk,
                        'cheat_probability': cheat_probability,
                        'global_cheat': global_cheat
                    }
            else:
                # Reset head_pose when no proper face detected
                head_pose.x = 0
                head_pose.y = 0
                head_pose.X_AXIS_CHEAT = 0
                head_pose.Y_AXIS_CHEAT = 0
        
        except Exception as e:
            cv2.putText(image, f"Pose Error: {str(e)[:30]}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Reset head_pose on error
            head_pose.x = 0
            head_pose.y = 0
            head_pose.X_AXIS_CHEAT = 0
            head_pose.Y_AXIS_CHEAT = 0
        
        return None

def create_progress_bar(value, max_value, label, color_class="normal"):
    """Create a custom progress bar"""
    percentage = min((value / max_value) * 100, 100)
    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: 500;">{label}</span>
            <span style="font-weight: 600;">{value:.1f}/{max_value}</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill progress-{color_class}" style="width: {percentage}%;"></div>
        </div>
    </div>
    """

def create_metric_card(title, value, subtitle="", status="normal"):
    """Create a custom metric card"""
    status_class = f"status-{status}"
    icon = "✅" if status == "normal" else "⚠️" if status == "warning" else "🚨"
    
    return f"""
    <div class="metric-card {status_class}">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; opacity: 0.9;">{title}</h3>
                <h2 style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{value}</h2>
                {f'<p style="margin: 0.25rem 0 0 0; opacity: 0.8; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ''}
            </div>
            <div style="font-size: 2rem;">{icon}</div>
        </div>
    </div>
    """

def create_live_monitoring_chart():
    """Create live monitoring chart - EXACT replica of detection.py graph from run.py"""
    try:
        # Use detection.py's actual YDATA like run.py
        ydata = getattr(detection, 'YDATA', [0] * 200)
        xdata = getattr(detection, 'XDATA', list(range(200)))
        
        # If YDATA doesn't exist, create it from our stored data
        if not hasattr(detection, 'YDATA') or len(ydata) == 0:
            # Initialize detection.py variables if they don't exist
            detection.XDATA = list(range(200))
            detection.YDATA = [0] * 200
            xdata = detection.XDATA
            ydata = detection.YDATA
        
        # Create matplotlib figure - EXACT replica of detection.py
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the red line exactly like detection.py: 'r-'
        line = ax.plot(xdata, ydata, 'r-')[0]
        
        # Set limits exactly like detection.py
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 1)
        
        # Set title and labels exactly like detection.py
        ax.set_title("SUSpicious Behaviour Detection", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Cheat Probablity", fontsize=12)  # Keep the typo like original
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"Error creating detection.py style chart: {e}")
        return None

def update_monitoring_data(risk_value):
    """Update live monitoring data - Use existing detection.py YDATA update"""
    try:
        # Use file-based state instead of st.session_state to avoid ScriptRunContext errors
        state_file = "proctoring_state.json"
        current_time = datetime.now()
        
        # *** USE EXISTING DETECTION.PY YDATA UPDATE LOGIC ***
        # The detection.py process() function already updates YDATA automatically
        # Just ensure YDATA is initialized
        if not hasattr(detection, 'YDATA'):
            detection.YDATA = [0] * 200
        if not hasattr(detection, 'XDATA'):
            detection.XDATA = list(range(200))
        
        # Update graph data using existing detection.py logic (done by process())
        # YDATA is updated in detection.run_detection() with: YDATA.pop(0); YDATA.append(PERCENTAGE_CHEAT)
        
        # No debug print to keep terminal clean
        
        # Read existing state for dashboard
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
            except:
                state = {}
        else:
            state = {}
        
        # Initialize history arrays if they don't exist
        if 'risk_history' not in state:
            state['risk_history'] = []
        if 'time_history' not in state:
            state['time_history'] = []
        
        # Add new data for dashboard history
        state['risk_history'].append(risk_value)
        state['time_history'].append(current_time.isoformat())
        
        # Keep only last 300 points for dashboard history
        if len(state['risk_history']) > 300:
            state['risk_history'] = state['risk_history'][-300:]
            state['time_history'] = state['time_history'][-300:]
        
        # Add real-time statistics
        state['current_risk'] = risk_value
        state['average_risk'] = sum(state['risk_history']) / len(state['risk_history'])
        state['max_risk'] = max(state['risk_history'])
        state['min_risk'] = min(state['risk_history'])
        state['last_updated'] = current_time.isoformat()
        
        # Calculate trend
        if len(state['risk_history']) >= 10:
            recent_avg = sum(state['risk_history'][-10:]) / 10
            older_avg = sum(state['risk_history'][-20:-10]) / 10 if len(state['risk_history']) >= 20 else recent_avg
            state['trend'] = 'increasing' if recent_avg > older_avg else 'decreasing'
        else:
            state['trend'] = 'stable'
        
        # Save back to file
        with open(state_file, 'w') as f:
            json.dump(state, f)
            
    except Exception as e:
        # Silently ignore errors to avoid breaking video processing
        print(f"Monitoring update error: {e}")
        pass

def create_risk_gauge(risk_value):
    """Create a risk level gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Level"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        font=dict(family="Inter", size=10),
        height=200,  # Reduced from 300 to 200
        margin=dict(l=10, r=10, t=40, b=10)  # Reduced margins
    )
    
    return fig

def main():
    # Start the detection modules like run.py (but as background threads)
    if 'detection_threads_started' not in st.session_state:
        st.session_state.detection_threads_started = True
        
        # Initialize detection.py variables
        detection.XDATA = list(range(200))
        detection.YDATA = [0] * 200
        detection.PERCENTAGE_CHEAT = 0
        detection.GLOBAL_CHEAT = 0
        
        # Note: Detection processing will be handled by VideoProcessor threads
        # which call detection.process() directly for each frame
        
        print("Started detection modules like run.py")
    
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Parallax AI</h1>
        <div class="live-indicator">
            <div class="live-dot"></div>
            LIVE MONITORING
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera Setup Instructions (inspired by graph.py user guidance)
    with st.expander("📹 Camera Setup Guide - Fix 'No Face Detected'", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Positioning:**
            - Sit directly in front of camera
            - Keep face centered in frame
            - Maintain 18-24 inches distance
            - Look directly at camera
            """)
        
        with col2:
            st.markdown("""
            **💡 Lighting:**
            - Ensure good lighting on face
            - Avoid backlighting from windows
            - Use room lighting or desk lamp
            - Face should be clearly visible
            """)
        
        with col3:
            st.markdown("""
            **🔧 Technical:**
            - Camera resolution: 720p minimum
            - Check camera permissions
            - Close other camera apps
            - Restart browser if needed
            """)
    
    # System Status Check
    try:
        import mediapipe as mp
        st.success("✅ MediaPipe System Operational - Enhanced monitoring active!")
    except Exception as e:
        st.error(f"❌ MediaPipe Error: {e}")
    
    # Add clear data button
    col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 2])
    with col_clear1:
        if st.button("🗑️ Clear Violation Logs"):
            clear_violation_logs()
            st.success("Violation logs cleared!")
            st.rerun()
    
    # Main Layout
    col1, col2 = st.columns([2.2, 1.8])
    
    with col1:
        st.markdown("### 🎥 Live Camera Feed")
        
        # Enhanced video container
        video_container = st.container()
        with video_container:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            
            # WebRTC streamer with enhanced config
            webrtc_ctx = webrtc_streamer(
                key="enhanced-proctoring-system",
                video_processor_factory=EnhancedProctoringProcessor,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "frameRate": {"ideal": 30}
                    },
                    "audio": False
                },
                async_processing=True,
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Connection status with enhanced styling
        if webrtc_ctx.state.playing:
            st.markdown("""
            <div class="custom-alert alert-success">
                <strong>🟢 Camera Active</strong> - AI monitoring in progress with real-time analysis
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-alert alert-warning">
                <strong>⚪ Camera Inactive</strong> - Click START to begin enhanced monitoring
            </div>
            """, unsafe_allow_html=True)
        
        # Live Suspicious Behavior Detection - moved here below video
        st.markdown("### 📊 Live Suspicious Behavior Detection")
        
        # Create a container for the live chart with fixed height
        chart_container = st.container()
        with chart_container:
            live_chart = create_live_monitoring_chart()
            if live_chart:
                st.pyplot(live_chart)  # Use st.pyplot for matplotlib instead of st.plotly_chart
            else:
                # Create a placeholder chart to show that monitoring is ready
                current_state = load_state()
                if current_state.get('monitoring_active', False):
                    # Check if we have any risk data
                    risk_history = current_state.get('risk_history', [])
                    cheat_prob = current_state.get('percentage_cheat', 0)
                    if len(risk_history) > 0:
                        st.info(f"📊 Live monitoring active - {len(risk_history)} data points collected. Current cheat probability: {cheat_prob:.3f}")
                    else:
                        st.info("📊 Live monitoring active - Collecting initial data for graph display...")
                else:
                    st.info("📊 Start camera to begin live behavior detection monitoring...")
    
    with col2:
        st.markdown("### 📊 Live Dashboard")
        
        # Load current state AND real-time module values for coordinated dashboard
        current_state = load_state()
        
        # Get real-time values from modules (like run.py dashboard coordination)
        real_time_x = getattr(head_pose, 'x', current_state.get('x_angle', 0))
        real_time_y = getattr(head_pose, 'y', current_state.get('y_angle', 0))
        real_time_x_cheat = getattr(head_pose, 'X_AXIS_CHEAT', 0)
        real_time_y_cheat = getattr(head_pose, 'Y_AXIS_CHEAT', 0)
        real_time_audio_cheat = getattr(audio, 'AUDIO_CHEAT', 0)
        real_time_audio_amplitude = getattr(audio, 'SOUND_AMPLITUDE', current_state.get('audio_level', 0))
        real_time_cheat_percentage = getattr(detection, 'PERCENTAGE_CHEAT', current_state.get('risk', 0))
        real_time_global_cheat = getattr(detection, 'GLOBAL_CHEAT', 0)
        
        # Use real-time values for dashboard (coordinated with video processing)
        dashboard_state = {
            'x_angle': real_time_x,
            'y_angle': real_time_y,
            'risk': real_time_cheat_percentage,
            'violation': real_time_global_cheat == 1,
            'audio_level': real_time_audio_amplitude,
            'total_violations': current_state.get('total_violations', 0),
            'frame_count': current_state.get('frame_count', 0),
            'timestamp': current_state.get('timestamp', datetime.now().isoformat()),
            'session_start': current_state.get('session_start', datetime.now().isoformat()),
            'x_axis_cheat': real_time_x_cheat,
            'y_axis_cheat': real_time_y_cheat,
            'audio_cheat': real_time_audio_cheat,
            'global_cheat': real_time_global_cheat
        }
        
        # Add refresh info and manual controls
        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col1:
            st.caption(f"🔄 Last update: {datetime.now().strftime('%H:%M:%S')} | Live data from modules")
        with refresh_col2:
            if st.button("🔄 Refresh", key="refresh_dashboard"):
                st.rerun()
        
        # Session Info (using coordinated state)
        try:
            session_duration = datetime.now() - datetime.fromisoformat(dashboard_state.get('session_start', datetime.now().isoformat()))
            duration_str = str(session_duration).split('.')[0]
        except:
            duration_str = "00:00:00"
        
        # Enhanced Status Cards (using real-time coordinated values)
        if dashboard_state['violation']:
            status = "violation"
            status_text = "VIOLATION ACTIVE"
            risk_color = "danger"
        elif dashboard_state['risk'] > 0.3:
            status = "warning"
            status_text = "ELEVATED RISK"
            risk_color = "warning"
        else:
            status = "normal"
            status_text = "ALL CLEAR"
            risk_color = "normal"
        
        # Primary Status Card with real-time data
        st.markdown(create_metric_card(
            "Current Status", 
            status_text,
            f"Risk: {dashboard_state['risk']:.1%}",
            status
        ), unsafe_allow_html=True)
        
        # Metrics Row - All metrics in one row with real-time values
        st.markdown("#### 📊 Live Metrics (Real-time)")
        
        # Add real-time status indicators
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        with status_col1:
            status_icon = "🟢" if abs(real_time_x) > 0.1 else "🟡"
            st.caption(f"{status_icon} Head X Active")
        with status_col2:
            status_icon = "🟢" if abs(real_time_y) > 0.1 else "🟡" 
            st.caption(f"{status_icon} Head Y Active")
        with status_col3:
            status_icon = "🟢" if real_time_audio_amplitude > 1 else "🟡"
            st.caption(f"{status_icon} Audio Active")
        with status_col4:
            status_icon = "🟢" if real_time_cheat_percentage > 0.01 else "🟡"
            st.caption(f"{status_icon} Detection Active")
        
        # Create 4 columns for all metrics in one row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Head X°", f"{dashboard_state['x_angle']:.1f}°", 
                     delta=f"Cheat: {dashboard_state['x_axis_cheat']}")
        
        with metric_col2:
            st.metric("Head Y°", f"{dashboard_state['y_angle']:.1f}°",
                     delta=f"Cheat: {dashboard_state['y_axis_cheat']}")
        
        with metric_col3:
            st.metric("Audio", f"{dashboard_state['audio_level']:.1f}",
                     delta=f"Cheat: {dashboard_state['audio_cheat']}")
            
        with metric_col4:
            st.metric("Total Violations", dashboard_state['total_violations'])
        
        # Real-time Detection Status
        st.markdown("#### 🔍 Detection Status")
        detection_col1, detection_col2, detection_col3 = st.columns(3)
        
        with detection_col1:
            st.metric("Cheat %", f"{dashboard_state['risk']:.3f}")
        
        with detection_col2:
            st.metric("Global Cheat", dashboard_state['global_cheat'])
        
        with detection_col3:
            st.metric("Session Time", duration_str)
        
        # Risk Level Gauge - Compact version with real-time data
        st.markdown("#### Risk Analysis")
        if dashboard_state['risk'] > 0:
            risk_fig = create_risk_gauge(dashboard_state['risk'])
            st.plotly_chart(risk_fig, use_container_width=True, config={'displayModeBar': False})
        
        # Progress Bars for monitoring levels with real-time values
        st.markdown("#### Monitoring Levels")
        st.markdown(create_progress_bar(
            abs(dashboard_state['x_angle']), 30, "Head X Movement", 
            "danger" if abs(dashboard_state['x_angle']) > 20 else "normal"
        ), unsafe_allow_html=True)
        
        st.markdown(create_progress_bar(
            abs(dashboard_state['y_angle']), 30, "Head Y Movement",
            "danger" if abs(dashboard_state['y_angle']) > 20 else "normal"
        ), unsafe_allow_html=True)
        
        # Audio Level with real-time value
        st.markdown(create_progress_bar(
            dashboard_state['audio_level'], 100.0, "Audio Level",
            "danger" if dashboard_state['audio_level'] > 20 else "normal"
        ), unsafe_allow_html=True)
        
        # System Information with real-time status
        with st.expander("📊 System Information", expanded=False):
            st.write(f"**Frame Count:** {dashboard_state['frame_count']:,}")
            st.write(f"**Last Update:** {dashboard_state['timestamp'][11:19]}")
            st.write(f"**Status:** {'🟢 Active' if webrtc_ctx.state.playing else '⚪ Inactive'}")
            st.write(f"**Violations Today:** {dashboard_state['total_violations']}")
            st.write(f"**Real-time X Cheat:** {dashboard_state['x_axis_cheat']}")
            st.write(f"**Real-time Y Cheat:** {dashboard_state['y_axis_cheat']}")
            st.write(f"**Real-time Audio Cheat:** {dashboard_state['audio_cheat']}")
            st.write(f"**Real-time Global Cheat:** {dashboard_state['global_cheat']}")
            st.write(f"**Detection Cheat %:** {dashboard_state['risk']:.6f}")
            
            # Real-time statistics (inspired by graph.py data analysis)
            try:
                state_file = "proctoring_state.json"
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        live_state = json.load(f)
                    
                    if 'current_risk' in live_state:
                        st.markdown("#### 📊 Live Analytics (Graph.py Enhanced)")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Risk", f"{live_state['current_risk']:.1%}")
                            st.metric("Average Risk", f"{live_state.get('average_risk', 0):.1%}")
                        
                        with col2:
                            st.metric("Max Risk", f"{live_state.get('max_risk', 0):.1%}")
                            st.metric("Min Risk", f"{live_state.get('min_risk', 0):.1%}")
                        
                        with col3:
                            trend = live_state.get('trend', 'stable')
                            trend_emoji = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
                            st.metric("Trend", f"{trend_emoji} {trend.title()}")
                            risk_history = live_state.get('risk_history', [])
                            st.metric("Data Points", f"{len(risk_history)}")
            except:
                pass

    # End of column layout - Now add Violation Analytics at the bottom
    st.markdown("---")
    st.markdown("### 📈 Violation Analytics")
    
    violations_data = load_violations_log()
    
    # Only show violations if monitoring has been active
    if violations_data and current_state.get('monitoring_active', False):
        # Create a container to prevent horizontal scrolling
        violations_container = st.container()
        with violations_container:
            # Recent violations table
            col_hist1, col_hist2 = st.columns([1.5, 1])
            
            with col_hist1:
                st.markdown("#### Recent Violations")
                recent_violations = violations_data[-10:]  # Last 10 violations
                
                df_violations = pd.DataFrame(recent_violations)
                df_violations['time'] = pd.to_datetime(df_violations['timestamp']).dt.strftime('%H:%M:%S')
                df_violations = df_violations[['time', 'type', 'risk']].rename(columns={
                    'time': 'Time', 'type': 'Violation Type', 'risk': 'Risk Level'
                })
                df_violations['Risk Level'] = df_violations['Risk Level'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(df_violations, use_container_width=True, hide_index=True)
            
            with col_hist2:
                st.markdown("#### Violation Summary")
                violation_types = pd.DataFrame(violations_data)['type'].value_counts()
                
                fig_pie = px.pie(
                    values=violation_types.values, 
                    names=violation_types.index,
                    title="Violation Types Distribution"
                )
                fig_pie.update_layout(
                    font=dict(family="Inter", size=10),
                    height=300,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No violations recorded yet. Monitoring is active and ready.")
    
    # Auto-refresh mechanism with enhanced performance
    if webrtc_ctx.state.playing:
        time.sleep(0.5)  # Faster refresh rate for better real-time experience
        st.rerun()

if __name__ == "__main__":
    main()