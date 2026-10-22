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

---

## 9. Opt-in Runtime Candidates and Full-State Validation Calibration (2026-10-10)

### Scope and isolation
- User approved posture/temporal corrections, bounded short-gap history retention, and validation-only full-runtime calibration. Do not interpret this as permission to promise 100% field recall or activate a new model automatically.
- Active checkpoint/scaler and server default remain the legacy detector at 0.90. `src/inference.py` only adds an overridable geometry hook and optional `frame_size`; legacy decisions remain reproducible. Shared `src/yolo_detector.py` was not modified, preserving extraction provenance.
- New `src/runtime_candidate.py` contains frozen profiles: `geometry_v2`, `gap_v2`, `temporal_v2`. The original `legacy_v1` remains a control. No further NN training or feature dimension change was needed: brief gaps retain the raw zero rows already present in training rather than invented/interpolated keypoints.
- `geometry_v2` measures equal-scale image-plane pixel geometry and honors the configured angle, aspect and signed torso-height thresholds. Arrays/live objects use the same confidence cutoff; missing shoulder/hip anchors or degenerate geometry are unknown, not upright evidence. This does not calibrate physical 3D angles or fix fisheye/perspective/capture resizing.
- `gap_v2` retains compatible gaps up to 3 frames and 0.30 s, with sufficiently visible previous/current hips and bounded center displacement. Long/incompatible gaps, >0.50 s stream interruption, resolution changes or explicit `target_changed` discard motion evidence and rewarm actual frames, not 30 copies of the returning pose. Fast continuous downward motion alone is no longer a target-switch reset. Confirmed fall holds survive visual discontinuities; explicit new identity resets the detector latch.
- `temporal_v2` adds three consecutive raw positive observations, five-frame low-score upright recovery, and a timestamped nose + both-hip descending/rising gate. Reliable motion anchors are required to override upright; motion alone cannot create a Fall. Votes precede geometric veto, so previous positive scores during a transition can confirm on the first gate-permitted frame. Missing poses reset votes and never invoke the model or begin new alerts; pose reacquisition still requires three observations.
- Hip compatibility is not identity tracking. The candidate API accepts explicit `target_changed`, but current live YOLO does not supply a reliable person-ID signal to the server. Limit initial supervised trials to one person. Camera ego-motion can confound the temporal gate.

### Calibration and frozen results
- `scripts/calibrate_runtime.py` replays the full state machine for all four predefined profiles and thresholds 0.50 through 0.99 on **validation only**, using the checkpoint-bound 50-video manifest. Only identical hashed neural inputs/predictions are memoized; gate/vote/recovery/latch decisions are rerun at every point. Tests verify cache/direct replay equivalence and fail on input mismatch.
- Predeclared objective: maximum **fresh alert-onset recall inside the original Fall annotation**, then minimum false ADL videos/episodes, minimum premature episodes, minimum median onset delay, highest threshold, simpler profile. A false alert latched from before a Fall is not credited as a true onset; late alerts remain separate. No interval stretching and no test-based parameter selection.
- Frozen local audit: `results/logs/runtime_v2/calibration.json`. Exclusive creation prevents overwriting selections; checkpoint, pose policy, feature/source hashes, code hashes, annotations and validation manifest are recorded. Model-forward calls per profile: 2,916 legacy/geometry and 2,863 gap/temporal, reused safely for the 50 threshold points.

| YOLO11m candidate profile | Validation-selected threshold | Fresh-onset interval confusion [[TN,FP],[FN,TP]] | Median onset delay |
|---|---:|---|---:|
| legacy_v1 | 0.86 | [[30,1],[1,18]] | 1.10 s |
| geometry_v2 | 0.78 | [[30,1],[1,18]] | 0.90 s |
| gap_v2 | 0.78 | [[29,2],[1,18]] | 0.90 s |
| **temporal_v2** | **0.72** | **[[30,1],[1,18]]** | **0.70 s** |

- Global selection was `temporal_v2` / 0.72, with 18/19 validation Fall onsets and 1/31 false ADL video. The missed validation clip `old:Fall_ke_depan_8` has 59/76 missing accepted poses and zero neural evaluations in this replay; a score threshold cannot repair absent visual evidence. The false ADL was controlled lying (`old:ADL_Berbaring_controlled_2`). ADL validation exposure is only 193 seconds; false-alert/hour values are explicitly a short-clip proxy, not continuous monitoring estimates.

### Regression after freezing selection
- `scripts/compare_runtime_candidates.py` verifies calibration integrity before/after evaluation, uses identical regression video IDs for all scenarios, rejects overlap with validation, and checks active artifact hashes. The first orchestration attempt exposed a relative/absolute report-path bug after baseline evaluation; it was fixed in the comparison script without changing runtime code or validation selection. The complete comparison then exited successfully.
- Tracked summary: `results/runtime_v2_comparison.json`. Detailed six replay reports and logs: `results/logs/runtime_v2/` (ignored). All table entries use fresh alert onsets inside the same annotations; no pre-fall onsets occurred.

| Classifier/runtime | Threshold | Interval Fall onsets | False ADL videos | Median delay |
|---|---:|---:|---:|---:|
| Active original / legacy | 0.90 | 18/19 | 1/31 | 1.17 s |
| YOLO11m candidate / legacy | 0.90 | 17/19 | 0/31 | 1.07 s |
| YOLO11m candidate / geometry | 0.90 | 18/19 | 1/31 | 0.73 s |
| YOLO11m candidate / gap | 0.90 | 18/19 | 1/31 | 0.73 s |
| YOLO11m candidate / temporal | 0.90 | 18/19 | 0/31 | 0.80 s |
| **YOLO11m candidate / validation-selected temporal** | **0.72** | **19/19** | **1/31** | **0.73 s** |

- Gap/temporal at 0.90 each have one late-only alert (still an interval miss). Selected 0.72 has no late-only or premature episodes. **Do not claim 100% real-world recall:** these 19 falls were already inspected to diagnose prior runtime failures; this is a regression set, not a pristine unseen final test. Validation itself is still 18/19. All thresholds/settings remained unchanged after regression results were seen.
- Relative to the YOLO11m legacy candidate, the selected runtime gains two detected falls while adding one false ADL clip. Compared with the original active baseline it gains one fall with the same one false ADL clip. Small correlated video-disjoint samples do not establish significance or unseen-person/scene performance.

### Controlled access and verification
- Server opt-in only: set `HAFIBOT_FALL_RUNTIME_CALIBRATION` to the frozen audit path and restart. `src/runtime_loader.py` refuses changed code/settings, checkpoint or pose-extraction signature. With the variable unset, it constructs the original `FallDetector` and original threshold. `.env.example` only contains a commented example; no active configuration was changed.
- Server passes the actual decoded frame dimensions. Replay uses source time at 15 FPS; live network timing/frame loss and mobile-camera ego-motion are not simulated. External motor ESTOP latching, caregiver push delivery, capture-to-alert latency and multi-person tracking remain outside detector replay.
- Both active artifact hashes remained exactly those in section 8. Candidate scalers are embedded in their checkpoint and were independently verified. Existing unrelated `data/monitoring/reminders.json` edits were retained byte-for-byte: legacy chatbot integration tests advanced prayer dates, and the exact pre-test state/hash was restored rather than replacing user changes with Git's version.
- Full suite: **136 passed**, three existing deprecation warnings; syntax compilation and `git diff --check` passed. Calibration and complete comparison processes exited normally. Changes from this runtime task have **not been committed or pushed**.
- Next required evidence before promotion: fresh subject/room-disjoint recordings; difficult intentional lying/sitting/bending hard negatives; pose-visibility/low-angle occlusion audit; long continuous ADL stationary and moving-robot trials; real capture/network alert latency and full notification/ESTOP delivery. Keep any future model/profile tuning separated from the new final holdout.

---

## 10. Summary of Progress, Runtime Evaluation, and Next Steps (October 2026)

### Key Accomplishments & State of the Project
1. **Retrained YOLO11m-Compatible 1D CNN-BiLSTM Candidate**:
   - Re-extracted all 332 dataset videos (29,588 frames) with YOLO11m-pose to eliminate the mismatch between training (formerly YOLOv8n) and deployment (YOLO11m-pose).
   - Provenance tracking with SHA-256 caching verified under `data/processed_yolo11m/`.
   - Candidate model trained and evaluated; checkpoint stored at `models/candidates/yolo11m/best_model.pt`.

2. **Runtime Candidate Enhancements (`src/runtime_candidate.py`)**:
   - **`geometry_v2`**: Pixel-aspect-ratio corrected torso inclination angles and bounding box ratios using actual frame pixel dimensions.
   - **`gap_v2`**: Bounded gap history preservation (<= 3 frames / 0.30 s) with reliable hip anchors, preventing artificial velocity spikes upon target re-entry without hallucinating keypoints.
   - **`temporal_v2`**: Multi-frame temporal voting (3 consecutive positive frames), vertical downward motion gating (head + hip descending), and 5-frame stable upright recovery.

3. **Validation Calibration & Regression Replay Results**:
   - Swept thresholds from 0.50 to 0.99 across all profiles using held-out validation only.
   - Selected operating point: **`temporal_v2` at threshold 0.72** (18/19 validation falls detected, 1/31 false ADL video, 0.70 s median delay).
   - On the 50 regression clips, `temporal_v2` detected **19/19 fall onsets** with 1/31 false ADL clips (median delay 0.73 s), compared to the active baseline legacy at 0.90 (18/19 falls, 1/31 false ADL, 1.17 s median delay).

4. **Safety & Deployment Status**:
   - **The candidate model remains inactive by default.** The active production model is `models/best_model.pt` with legacy runtime and 0.90 threshold.
   - Added an opt-in switch via the `HAFIBOT_FALL_RUNTIME_CALIBRATION` environment variable through `src/runtime_loader.py` for safe, supervised evaluation.
   - Full test suite verified with 136 passing tests.

5. **Motor Dynamics & Human Following Enhancements**:
   - Diagnosed 4WD 1.7 kg skid-steer static friction and L298N forward voltage drop (~2-3V) causing motor stalls.
   - Defined breakaway kick-start pulse (PWM 255 for 100 ms) and deadband floor remapping (180-255 PWM) in motor controller architecture.
   - Implemented updated PWM thresholds in Arduino firmware (`hafibot_motor_controller.ino`): `SPEED_FORWARD_MIN = 180`, `SPEED_FORWARD_MAX = 255`, `KICKSTART_FEEDFORWARD_PWM = 255` (150 ms duration), and instant ramp deadband jump to eliminate in-air motor cogging/stalls.

### Recommended Next Steps
- **Supervised Robot Trials**: Run live trials with the camera feed and robot using the opt-in variable to test natural interaction (sitting, walking, bending, floor transfers) before promoting the candidate.
- **Continuous Multi-Hour ADL Benchmark**: Collect extended continuous ADL recordings to measure the true false-alarm rate per hour.
- **Low-Angle Occlusion Audit**: Investigate remaining edge cases where pose keypoints drop below threshold due to the 21 cm ground-level camera perspective.
