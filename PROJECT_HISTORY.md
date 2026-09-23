# Hafibot Project Context & Technical History Handover

This document serves as a comprehensive, structured briefing designed for both human review and immediate ingestion by an AI coding assistant in future sessions.

---

## 1. Project Overview & System Identity
- **System Name**: Hafibot (Smart Elderly-Care Mobile Companion Robot).
- **Hardware Stack**:
  - Edge Computing: NVIDIA Jetson Nano Developer Kit (4GB, Tegra X1).
  - Low-level Controller: Arduino Mega 2560 Pro Mini (DC motors with optical encoders + 3x HC-SR04 ultrasonic sensors).
  - Vision: CSI Camera (IMX219 sensor, 160 deg Wide-Angle FOV).
  - Audio: MAX98357A I2S 3W D-Class amplifier + speaker + USB microphone.
  - Host AI Server: Laptop / Workstation with NVIDIA GPU (CUDA acceleration).
- **Active Clean Workspace**: `D:\Tugas Akhir\hafibot`
- **Legacy Workspace (Reference)**: `D:\Tugas Akhir\Fall Detection 1D CNN-LSTM`
- **GitHub Remote**: `https://github.com/udeanfauzan/hafibot.git` (branch: `main`).

---

## 2. Distributed Architecture (Crucial Paradigm Shift)

### Why Edge Inference Was Replaced by Distributed AI Server:
- **Jetson Nano Limitations**: The Jetson Nano (4GB) could not sustainably run concurrent MediaPipe/YOLO-Pose inference, 1D CNN-LSTM evaluation, Flask streaming, ROS nodes, and Voice Chatbot without hitting nvarguscamerasrc timeouts, GPU memory exhaustion, or significant video feed latency.
- **Distributed Solution**:
  - **Edge Robot (Jetson Nano + Arduino Mega)**:
    - Lightweight edge agent (`scripts/jetson_edge_agent.py`) captures CSI camera frames (640x480 @ 15 FPS JPEG) and streams them over LAN/Tailscale to the laptop server.
    - Relays serial commands `<M,speed,steer,flag>` to the Arduino motor controller.
    - Manages local speaker alert playback (`--enable-voice --speaker-device default`).
  - **Host AI Server (Laptop GPU)**:
    - Runs `src/server_app.py` on port 5000.
    - Performs pose estimation (YOLO11m-pose), 1D CNN-BiLSTM fall inference, geometric posture verification, adaptive PID human following calculations, voice chatbot service, and Web Push notifications.

---

## 3. Fall Detection Pipeline & Key Algorithmic Solutions

### 3.1 Neural Network Model
- **Input**: 30-frame temporal window (2.0 seconds at 15 FPS) of normalized 2D skeleton keypoints.
- **Model Architecture**: 1D Convolutional feature extractor + Bidirectional LSTM temporal dynamics + Self-Attention mechanism.
- **Active Weights**: `models/best_model.pt` with `models/scaler_params.npz`.

### 3.2 Evolution of Pose Estimators:
1. **MediaPipe Pose 0.8.5**: Initial setup; CPU bound on Windows and struggled with inverted/lying postures.
2. **YOLOv8n-pose**: Lightweight edge test, but showed jitter and false negatives during backward falls.
3. **YOLO11m-pose (`models/yolo11m-pose.pt`)**: Current active detector on GPU server. High keypoint stability even during rapid drops and occlusions.

### 3.3 Critical Problem 1: False Alarms When Leaving Camera Frame
- **Problem**: When a person stepped out of view, the inference buffer filled with zero/noisy values, causing fall probability to spike to 99% upon detecting any returning pose.
- **Solution**: Pose-loss handling logic in `src/inference.py`:
  - Does not evaluate the model when no pose is visible.
  - Forces displayed fall confidence to 0% during missing frames.
  - Requires 3 consecutive detected frames (~0.2s) before confirming state transitions upon re-entry.

### 3.4 Critical Problem 2: False Alarms on Sitting & Bending
- **Problem**: Walking away from the robot caused visual scale shrinkage and the low camera angle (21 cm off ground) triggered false positives even while sitting still.
- **Solution**: **Geometric Posture Gating** (`src/posture.py`):
  - Evaluates torso inclination angle relative to vertical axis using shoulder and hip keypoints.
  - A fall is **only** asserted if model probability > threshold **AND** posture is confirmed as lying/grounded (torso angle < 45 deg or low vertical torso span).
  - Sitting upright (torso angle ~80-90 deg) is strictly gated as ADL, eliminating sitting false alarms.

---

## 4. Autonomous Human Following & Motor Control

### 4.1 Visual Distance Estimation via Shoulder Width
- Distance is estimated using pixel span between left and right shoulders (`kpt 5` & `kpt 6`):
  - $\text{Distance} = \frac{f \times W_{\text{real}}}{W_{\text{pixel}}}$
- Includes a calibration tool (`scripts/calibrate_shoulder_distance.py`) and Web Dashboard recalibration slider (`data/monitoring/shoulder_calibration.json`) to adjust for differing shoulder widths.

### 4.2 Smooth PID Steering Controller (`src/human_following.py`)
- **Problem**: Robot abruptly jerked and overshot the target when turning, causing target loss.
- **Solution**:
  - Applied PID controller with damping on lateral offset.
  - Angular rate slew-limiting for smooth steering transitions.
  - Target distance deadband set to 100-180 cm.
  - Safety override: Sonar obstacle detection (< 25 cm) halts robot forward drive.

---

## 5. Dataset Lifecycle & Training History

1. **Initial Raw Dataset (`data/raw/`)**: 191 videos recorded at 26 cm camera height.
2. **Timing-Repaired Realtime July Dataset (`data/dataset new july realtime/`)**: 81 videos. Fixed a 2x slow-motion recording artifact down to authentic 15 FPS real-time motion (`scripts/repair_slow_motion_dataset.py`).
3. **Robot Perspective 21 cm Dataset (`data/new dataset 20 aug/`)**: 60 videos recorded at exactly 21 cm ground height (matching real robot chassis) across 3 subjects.
4. **Extracted Features**: Pre-extracted YOLO features ready for rapid training (`data/processed_yolo*/`).

---

## 6. Directory Layout of Clean Repo (`D:\Tugas Akhir\hafibot`)

```text
hafibot/
|-- arduino/hafibot_motor_controller/    # Arduino Mega firmware (PWM, Sonar, Protocol)
|-- data/                                # Video datasets, YOLO features, annotations
|-- env_falldetect/                      # Virtual environment junction
|-- img/                                 # UI web icons and branding
|-- models/                              # Trained weights (1D CNN-LSTM, YOLO11m, Whisper)
|-- results/                             # Evaluation charts, confusion matrices, demo JSONs
|-- robotchatbot/                        # Voice chatbot engine & models (DistilBERT + Qwen)
|-- scripts/                             # Utility & edge scripts (jetson_edge_agent.py, etc.)
|-- src/                                 # Main backend codebase (server_app, inference, etc.)
|-- start_server.bat                     # Windows 1-click server runner
|-- requirements.txt                     # Dependencies list
|-- pytest.ini                           # Test suite configuration
|-- README.md                            # Comprehensive documentation
`-- PROJECT_HISTORY.md                   # This handover brief
```

---

## 7. Operational Instructions for Future Sessions

- **Starting AI Server (Laptop)**:
  ```cmd
  cd "D:\Tugas Akhir\hafibot"
  start_server.bat
  ```
  Web UI: `http://localhost:5000` (User: `caregiver`, Pass: `hafibot123`).
- **Starting Edge Robot (Jetson Nano)**:
  ```bash
  python3 scripts/jetson_edge_agent.py --server http://<LAPTOP_IP>:5000 --enable-voice --speaker-device default
  ```
- **Running Verification Tests**:
  ```powershell
  cd "D:\Tugas Akhir\hafibot"
  .\env_falldetect\Scripts\activate
  pytest tests/test_posture.py tests/test_model_architecture.py tests/test_fall_alarm.py tests/test_feature_pipeline.py
  ```