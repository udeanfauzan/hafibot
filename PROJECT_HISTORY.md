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
3. **YOLO11m-pose (`models/yolo11m-pose.pt`)**: Current active detector on GPU server. This configuration does not guarantee continuous pose visibility; the October re-extraction found 635 missing-pose frames out of 29,588.

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
  - Current code blocks a positive model prediction when posture is classified as upright or the hip trajectory is rising; it does not require a separate positive lying confirmation in every case.
  - Upright gating reduces false alarms but can also suppress true falls. The October replay still found an ADL false alarm with the active model, so do not treat this as a false-alarm-free guarantee.

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

---

## 8. YOLO11m-Compatible Retraining Experiment (2026-10-10)

### Verified mismatch and implementation
- The active fall checkpoint's three extraction manifests identify **YOLOv8n-pose**, while the live server uses **YOLO11m-pose**. This retrains the **CNN-BiLSTM fall classifier**, not YOLO's pretrained pose weights.
- Offline extraction and live inference now share the configured YOLO11m weights, confidence 0.35, 640 input-size setting, selected 13 joints, target tracking, and anatomical rejection policy. Default deployment no longer silently falls back to a different pose model.
- New cache: `data/processed_yolo11m/{old,new_july,new_aug}/`. Old YOLOv8 caches remain untouched. Cache provenance binds weight/policy/source/array hashes; interrupted extraction is resumable, with atomic manifests and retries for transient Windows file locks.
- All **332 videos / 29,588 frames** were re-extracted and verified; 635 frames have no accepted pose. Counts: old 112 ADL + 79 Fall, July 53 ADL + 28 Fall, August 39 ADL + 21 Fall.
- Kept `legacy_v1` features/augmentation, 30-frame windows, 130 engineered features, CNN-BiLSTM architecture, seed 42, and video-stratified split. Video groups: 232 train / 50 validation / 50 test; eligible windows: 2,968 / 618 / 624. This is **not subject-disjoint** evaluation; August subject metadata remains incomplete.

### Training artifacts and evaluation
- Candidate: `models/candidates/yolo11m/best_model.pt`; paired scaler: `models/candidates/yolo11m/scaler_params.npz`. These are local, ignored binary artifacts, not included in Git.
- Training used the GTX 1650, ran 35 epochs, and stopped after 25 epochs without validation-loss improvement. Best epoch 10: validation loss ~0.1682, validation window accuracy 94.82%; final test argmax window accuracy 96.79% (not deployment-threshold or full-runtime accuracy).
- Threshold 0.93 was chosen on **validation classifier video maxima only**, before test evaluation. It is not a recommended deployment threshold based on this experiment.
- Compared both checkpoints on the same YOLO11m inputs and the same held-out 31 ADL + 19 Fall videos. Runtime replay uses actual `FallDetector` posture gates, pose-loss/reacquisition logic and latches, with synthetic 15 FPS source time. A detected Fall must overlap the annotated Fall interval.

| Checkpoint / threshold | Offline classifier video-max confusion [[TN,FP],[FN,TP]] | Runtime interval-replay confusion |
|---|---|---|
| Active baseline / 0.90 | [[27,4],[0,19]] | [[30,1],[1,18]] |
| YOLO11m candidate / 0.90 | [[30,1],[0,19]] | [[31,0],[2,17]] |
| YOLO11m candidate / validation 0.93 | [[30,1],[0,19]] | [[31,0],[4,15]] |

- At 0.90, the candidate removes the one runtime ADL false alarm but misses one additional Fall: **89.5% recall versus baseline 94.7%**. Do not promote it solely because offline video-max accuracy is 98%.
- At 0.93, two of the four interval misses produce alerts later in the clip, after the annotated interval. Report these as late alerts, not as clips with no alert at all.
- Diagnostic at 0.90: `old:Fall_ke_depan_9` has seven raw positive frames (33–39, max probability ~0.9919), all posture-gated, followed by substantial pose loss. `old:Fall_ke_belakang_3` has runtime maximum ~0.8999536, one missing pose at frame 47 and re-prime at 48. Do **not** tune the threshold on these test clips.
- Candidate tensors, embedded/external scaler pairing, finite history and disjoint video IDs were verified. Active checkpoint and scaler SHA-256 remain unchanged.
- **Candidate NOT activated.** Active fall weights remain `models/best_model.pt`, with threshold 0.90. Next work should calibrate/validate the full runtime state machine, audit pose-loss buffer resets and posture geometry, then test continuous ADL and controlled robot trials.

### Reproduction and caveats
- Run extraction/training as documented in README, then `python scripts/compare_pose_retraining.py`.
- Tracked summary: `results/yolo11m_retraining_comparison.json`. Detailed JSON, logs, history, plots and missed-Fall diagnostics: `results/logs/yolo11m_retrain/`.
- Full regression suite: **101 passed** (three existing dependency/fixture deprecation warnings). On this Windows environment use a fresh workspace-local pytest `--basetemp` and `-p no:cacheprovider` if the default temp directory denies access.
- Training completed its final evaluation and saved history, but the process then exited abnormally (`0xc0000409`, Windows event identified `ucrtbase.dll`). The root cause is unconfirmed. Subsequent independent checkpoint validation, calibration and all six comparison subprocesses exited normally; do not describe the training process itself as a clean exit.
- This is one training seed and a small correlated video-level holdout, not a significance test or field-reliability estimate. Replay excludes camera/network latency, unseen-person/room generalization, long-duration false alarms, and the external caregiver notification/motor-delivery chain.
