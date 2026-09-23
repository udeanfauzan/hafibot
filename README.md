# Hafibot - Smart Elderly-Care Mobile Robot

Hafibot adalah robot pendamping lansia cerdas berbasis mobile robot dengan arsitektur komputasi terdistribusi (Distributed AI Server + Edge Robot). Sistem ini mengintegrasikan **Fall Detection** (YOLOv8/11 Pose + 1D CNN-BiLSTM Attention), **Autonomous Human Following** berbasis PID, serta **Voice Chatbot Interaktif** (DistilBERT + Qwen).

---

## Arsitektur Sistem

Sistem dibagi menjadi dua komponen utama yang terhubung via LAN / Tailscale:

```text
+---------------------------------------------------------+
|                    EDGE MOBILE ROBOT                    |
|  - NVIDIA Jetson Nano (4GB)                             |
|  - CSI Camera (IMX219 160 deg FOV, 640x480 @ 15 FPS)    |
|  - Arduino Mega 2560 Pro (Motor Differential + Sonar)   |
|  - MAX98357A I2S Amplifier + Speaker (Voice & Beep)     |
|  - USB Microphone                                       |
|  Script: scripts/jetson_edge_agent.py                   |
+----------------------------+----------------------------+
                             |  HTTP / WebSocket / LAN
                             v
+---------------------------------------------------------+
|                 LAPTOP / WORKSTATION AI SERVER          |
|  - NVIDIA GPU Acceleration (CUDA)                       |
|  - YOLO Pose Estimator (YOLO11m / YOLOv8s)              |
|  - 1D CNN-BiLSTM Attention Fall Classifier (PyTorch)    |
|  - Geometric Posture Gating (Torso Angle & Lying Check) |
|  - Adaptive Human Following Controller (PID Distance)   |
|  - Voice Chatbot Engine (DistilBERT + Qwen)             |
|  - Web Monitoring Dashboard & Web Push Notifier (VAPID) |
|  Server: src/server_app.py (Port 5000)                  |
+---------------------------------------------------------+
```

---

## Struktur Direktori Repository

```text
hafibot/
|-- arduino/
|   `-- hafibot_motor_controller/    # Firmware Arduino Mega 2560 Pro (Motor & Sonar)
|-- data/
|   |-- raw/                         # Dataset video ADL & Fall awal (191 video)
|   |-- dataset new july realtime/   # Dataset 15 FPS real-time (81 video)
|   |-- new dataset 20 aug/          # Dataset sudut kamera robot 21 cm (60 video)
|   |-- processed_yolo*/             # Ekstraksi fitur koordinat YOLO (train-ready)
|   |-- monitoring/                  # Log insiden, snapshot, dan kalibrasi jarak
|   `-- fall_annotations*.json       # Anotasi temporal frame awal-akhir jatuh
|-- env_falldetect/                  # Virtual environment Python terisolasi
|-- img/                             # Aset visual & logo monitoring web
|-- models/
|   |-- best_model.pt                # Model aktif 1D CNN-LSTM Fall Detection
|   |-- scaler_params.npz            # Parameter normalisasi fitur
|   |-- yolo11m-pose.pt              # Model YOLO pose estimator akurasi tinggi
|   |-- yolov8n-pose.pt / .onnx      # Model YOLO pose ringan / edge
|   |-- distilbert-base-uncased/     # Model Transformer NLU chatbot
|   `-- whisper/base.pt              # Model Speech-to-Text Whisper
|-- results/
|   |-- figures/                     # Grafik evaluasi, confusion matrix, loss curves
|   `-- yolo11m_pose_*.json          # Contoh ekstraksi koordinat pose
|-- robotchatbot/                    # Layanan dan pipeline Voice Chatbot AI
|-- scripts/
|   |-- jetson_edge_agent.py         # Agent utama edge Jetson Nano (Stream & Gateway)
|   |-- test_arduino_motor.py        # Diagnostik mandiri komunikasi serial & motor
|   |-- setup_arduino_usb.sh         # Skrip izin port serial USB
|   |-- annotate_falls.py            # Tool GUI anotasi frame kejadian jatuh
|   `-- calibrate_shoulder_distance.py # Tool kalibrasi estimasi jarak lebar bahu
|-- src/
|   |-- server_app.py                # Server Flask Web Monitoring & Inference Hub
|   |-- inference.py                 # Pipeline inferensi Fall Detection real-time
|   |-- yolo_detector.py             # Wrapper YOLOv8 / YOLO11 Pose Estimation
|   |-- posture.py                   # Posture Analyzer & Geometric Verification
|   |-- human_following.py           # Logika kendali tracking orang berbasis PID
|   |-- model.py                     # Arsitektur Neural Network 1D CNN-BiLSTM Attention
|   |-- train.py                     # Skrip training model klasifikasi jatuh
|   `-- chatbot_service.py           # Integrasi chatbot suara ke monitoring
|-- tests/                           # Unit tests (Pytest)
|-- .env.example                     # Template konfigurasi environment
|-- requirements.txt                 # Dependensi Python
|-- start_server.bat                 # One-click launcher AI Server di Windows
`-- Vapid Keys.txt                   # Kunci Web Push Notification browser
```

---

## Panduan Menjalankan Sistem

### 1. Menjalankan AI Server di Laptop / Workstation

1. Pastikan dependencies terpasang atau gunakan virtual environment yang telah disediakan.
2. Jalankan server dengan mengklik ganda `start_server.bat` atau via terminal:
   ```powershell
   .\env_falldetect\Scripts\activate
   python src\server_app.py --host 0.0.0.0 --port 5000
   ```
3. Buka browser di `http://localhost:5000` (atau `http://<IP_LAPTOP>:5000`).
4. Login caregiver default:
   - **Username**: `caregiver`
   - **Password**: `hafibot123`
5. Izinkan Web Push Notification pada browser saat pop-up muncul.

### 2. Menjalankan Edge Agent di Jetson Nano Robot

1. Pastikan Arduino terhubung ke port USB Jetson Nano (`/dev/ttyUSB0` atau `/dev/ttyACM0`).
2. Jalankan edge streamer & gateway (sesuaikan IP laptop server / Tailscale IP):
   ```bash
   python3 scripts/jetson_edge_agent.py --server http://<IP_LAPTOP>:5000 --enable-voice --speaker-device default
   ```
3. Robot otomatis melakukan streaming kamera, menerima perintah gerak mengikuti pengguna (*human following*), dan membunyikan alarm suara jika terdeteksi kondisi darurat jatuh.

---

## Fitur Keamanan & Anti-False-Alarm

1. **Geometric Posture Gating**:
   - Model AI 1D CNN-LSTM dikombinasikan dengan verifikasi geometri postur tubuh (kemiringan sudut torso dan deteksi terbaring/di lantai).
   - Menghilangkan false alarm saat pengguna duduk tenang, membungkuk sesaat, atau berjalan menjauhi robot.
2. **Smooth PID Steering**:
   - Kontrol kemudi robot menggunakan kalkulasi PID halus untuk mencegah robot overshooting dan kehilangan target visual.
3. **Latched ESTOP on Fall**:
   - Ketika kejadian jatuh terverifikasi, robot langsung mengunci status darurat (*Emergency Stop*), menghentikan motor, memutar alarm peringatan, dan mengirim Web Push Notification seketika ke perangkat caregiver.

---

## Menjalankan Test Suite

Untuk memverifikasi integritas modul:
```powershell
pytest tests/
```
