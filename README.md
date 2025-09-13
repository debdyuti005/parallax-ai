# Parallax AI - Advanced Proctoring System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-green.svg)

**Parallax AI** is an advanced real-time proctoring system that uses computer vision and audio analysis to monitor user behavior during examinations or assessments.


## 🌟 Features

- **Real-time Head Pose Detection** - MediaPipe-based facial landmark tracking
- **Audio Monitoring** - Suspicious sound detection and analysis
- **Live Dashboard** - Real-time metrics and violation tracking
- **Interactive Web Interface** - Streamlit-based user-friendly interface
- **Violation Logging** - Comprehensive logging and analytics
- **Live Graph Visualization** - Real-time behavior analysis charts

## 🛠️ Technologies Used

- **Python 3.8+**
- **Streamlit** - Web interface
- **MediaPipe** - Face detection and pose estimation
- **OpenCV** - Computer vision processing
- **NumPy** - Numerical computations
- **Matplotlib** - Graph visualization
- **SoundDevice** - Audio capture and analysis

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam and microphone access
- Windows/Linux/macOS

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/debdyuti005/parallax-ai.git
   cd parallax-ai
   ```

2. **Create and activate virtual environment:**
   
   **Windows:**
   ```bash
   python -m venv parallax-ai
   .\parallax-ai\Scripts\activate
   ```
   
   **Linux/macOS:**
   ```bash
   python3 -m venv parallax-ai
   source parallax-ai/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run src/app.py
   ```

## 🚀 Usage

### Web Interface (Recommended)
```bash
streamlit run src/app.py
```
Then open your browser to `http://localhost:8501`

### Command Line (Original)
```bash
python src/run.py
```

## 📊 Features Overview

### 🎥 Live Camera Feed
- Real-time video processing with MediaPipe
- Head pose estimation and tracking
- Facial landmark detection

### 📈 Live Dashboard
- Real-time metrics display
- Violation count tracking
- Session duration monitoring
- Audio level indicators

### 🔍 Detection System
- **Head Movement Detection** - Monitors looking left/right/down
- **Audio Analysis** - Detects suspicious sounds
- **Cheat Probability Calculation** - Advanced algorithm combining multiple factors
- **Violation Logging** - Timestamped violation records

### 📊 Analytics
- Live behavior probability graph
- Historical violation data
- Session analytics and reporting

## 🏗️ Architecture

```
Parallax AI Architecture:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │    │   Audio Input   │    │  Web Interface  │
│   (MediaPipe)   │    │ (SoundDevice)   │    │   (Streamlit)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Detection Engine                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ head_pose.py│  │  audio.py   │  │     detection.py        │  │
│  │             │  │             │  │                         │  │
│  │ • X/Y Angle │  │ • Amplitude │  │ • Cheat Probability     │  │
│  │ • Cheat     │  │ • Threshold │  │ • Global Cheat Flag     │  │
│  │   Flags     │  │   Detection │  │ • Graph Data (YDATA)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │    Live Dashboard       │
                    │  • Real-time Metrics    │
                    │  • Violation Tracking   │
                    │  • Live Graph          │
                    │  • Session Analytics   │
                    └─────────────────────────┘
```

## 🔧 Configuration

### Camera Settings
- Resolution: 720p minimum recommended
- Frame rate: 30 FPS
- Lighting: Good front lighting required

### Detection Thresholds
- Head X movement: ±10° for violation
- Head Y movement: ±5° for violation  
- Audio threshold: 20 amplitude units
- Cheat probability threshold: 0.6

## 📝 File Structure

```
src/
├── app.py          # Main Streamlit web application
├── audio.py        # Audio detection and analysis
├── detection.py    # Core detection logic and algorithms
├── head_pose.py    # MediaPipe head pose estimation
├── run.py          # Original command-line version
└── ui.py           # Additional UI components

Data Files:
├── proctoring_state.json    # Session state data
├── violations_log.json      # Violation records
└── live_graph_data.json     # Graph data storage
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

**"No Face Detected"**
- Ensure good lighting on your face
- Position yourself centered in camera view
- Check camera permissions in browser

**Audio Not Working**
- Check microphone permissions
- Verify microphone is not muted
- Close other applications using microphone

**Performance Issues**
- Close unnecessary browser tabs
- Ensure good internet connection
- Check system resources (CPU/RAM)

## 📧 Support

For support and questions:
- Create an issue in this repository
- Contact: [debdyutimondal87@gmail.com]

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Enhanced analytics dashboard
- [ ] Mobile device support
- [ ] Integration with Learning Management Systems
- [ ] Advanced ML models for behavior detection
- [ ] Cloud deployment options

---

**Parallax AI** - Revolutionizing digital proctoring with advanced AI technology.