# Hafibot - Smart Elderly-Care Mobile Robot

Hafibot adalah robot pendamping lansia cerdas berbasis mobile robot dengan arsitektur komputasi terdistribusi (Distributed AI Server + Edge Robot). Sistem ini mengintegrasikan **Fall Detection** (YOLO11m Pose + 1D CNN-BiLSTM Attention), **Autonomous Human Following** berbasis PID, serta **Voice Chatbot Interaktif** (DistilBERT + Qwen).

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
|  - YOLO Pose Estimator (YOLO11m-pose)                  |
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
   - Memblokir kandidat alarm saat postur terverifikasi tegak; bukan jaminan bebas false alarm, dan gate yang terlalu ketat dapat melewatkan jatuh.
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

## Retraining dengan Pose Deployment (YOLO11m)

Model Fall lama dilatih dari fitur YOLOv8n, sementara server memakai YOLO11m.
Ekstraksi sekarang memakai konfigurasi dan pemilihan orang yang sama dengan
deployment, termasuk confidence 0.35, normalisasi koordinat, dan pemeriksaan
anatomi. Cache baru dipisahkan dari cache YOLOv8n; bobot, konfigurasi, dan kode
detektor dicatat dengan SHA-256 agar fitur berbeda tidak tercampur.

Jalankan dari root repository dengan environment Python aktif:

```powershell
python src/extract_features_yolo.py --all --batch-size 8
python src/train.py --checkpoint models/candidates/yolo11m/best_model.pt --scaler models/candidates/yolo11m/scaler_params.npz --history results/logs/yolo11m_retrain/training_history.npy
python scripts/calibrate_threshold.py --checkpoint models/candidates/yolo11m/best_model.pt --output results/logs/yolo11m_retrain/threshold_calibration.json
python src/evaluate.py --checkpoint models/candidates/yolo11m/best_model.pt --output results/logs/yolo11m_retrain/evaluation.json --figures-dir results/figures/yolo11m_retrain --history results/logs/yolo11m_retrain/training_history.npy
python scripts/evaluate_runtime_replay.py --checkpoint models/candidates/yolo11m/best_model.pt --output results/logs/yolo11m_retrain/runtime_replay.json
```

Fitur baru tersimpan di `data/processed_yolo11m/{old,new_july,new_aug}/`.
Ekstraksi dapat dilanjutkan dengan perintah yang sama; cache hanya digunakan
jika hash bobot, sumber video, dan fitur cocok. FPS sumber harus 15; jangan
mengubah metadata FPS untuk mengganti kecepatan gerakan. Jangan menjalankan
ekstraksi dan training secara bersamaan pada cache yang sama.

Gunakan `--threshold` pada evaluasi untuk nilai hasil kalibrasi validation.
Evaluasi classifier menggunakan window/video maxima. Runtime replay juga
menerapkan posture gating, pose reacquisition, dan latch pada waktu video,
tetapi tidak mengukur latency kamera/jaringan. Untuk membandingkan model lama
pada input deployment baru, gunakan `--checkpoint models/best_model.pt
--feature-root data/processed_yolo11m` pada perintah evaluasi.

Percobaan pertama mempertahankan arsitektur dan split per video yang sama.
Hasil ini belum membuktikan generalisasi pada orang atau ruangan baru.
Checkpoint kandidat dan scaler tidak mengganti model aktif secara otomatis;
pastikan hasil replay dan uji robot memadai sebelum aktivasi.
Default `src/train.py` juga menulis ke direktori kandidat agar model aktif
tidak tertimpa tanpa menentukan path output secara eksplisit.

Setelah training, seluruh kalibrasi dan perbandingan dapat dijalankan sekaligus:

```powershell
python scripts/compare_pose_retraining.py
```

Ringkasan tersedia di `results/yolo11m_retraining_comparison.json`; laporan,
log, dan grafik detail di `results/logs/yolo11m_retrain/`. Script memastikan
kedua model dievaluasi pada video test yang sama dan hash model aktif tetap
tidak berubah. Threshold kandidat dipilih pada validation sebelum test.

### Hasil percobaan 10 Oktober 2026

Ekstraksi YOLO11m selesai untuk **332 video / 29.588 frame**. Training berhenti
di epoch 35; checkpoint terbaik berdasarkan validation loss adalah epoch 10.
Pada 50 video test yang sama, replay **dengan gate dan latch aktif** memberi:

| Classifier (input YOLO11m) | Threshold | Fall terdeteksi dalam interval anotasi | Video ADL false alarm |
|---|---:|---:|---:|
| Model aktif lama | 0.90 | 18/19 (94.7%) | 1/31 |
| Kandidat retraining | 0.90 | 17/19 (89.5%) | 0/31 |
| Kandidat, threshold validation classifier | 0.93 | 15/19 (78.9%) | 0/31 |

Threshold 0.93 dipilih **hanya dari validation classifier video maxima**, bukan
dari test. Dua kasus di baris terakhir tetap menghasilkan alert setelah
interval anotasi berakhir; angka 15/19 memakai kriteria interval yang lebih
ketat daripada sekadar ada alert di dalam video.

Offline classifier video-max accuracy kandidat adalah 98%, tetapi itu **bukan
akurasi runtime**: kedua kandidat threshold mendeteksi semua 19 video Fall
secara offline, sedangkan posture gating dan perilaku buffer mengubah hasil
replay. Salah satu Fall yang terlewat memiliki tujuh prediksi positif yang
semuanya diblokir gate; kasus lain berada sedikit di bawah threshold 0.90 dan
mengalami reset buffer setelah satu frame pose hilang.

**Kandidat belum diaktifkan:** pengurangan false alarm disertai penurunan
recall. Model aktif dan scaler lama tetap dipertahankan. Perlu evaluasi
validation yang mengikuti state machine runtime, audit pose-loss/gating,
serta pengujian robot/ADL kontinu sebelum mengganti model. Hanya ada 19 kasus
Fall test dan split ini bukan subject-disjoint; jangan menyatakan reliabilitas
lapangan dari persentase tersebut.

## Kandidat Runtime: Geometri, Pose Dropout, dan Temporal Confirmation

**Eksperimental / opt-in, bukan jaminan recall 100%.** Default server tetap
memakai model aktif lama, threshold 0.90, serta runtime legacy. Bobot dan scaler
aktif tidak diganti. Profil kandidat dibekukan sebelum kalibrasi validation:

| Profil | Perubahan terhadap profil sebelumnya |
|---|---|
| `legacy_v1` | Baseline yang dapat direproduksi |
| `geometry_v2` | Sudut torso dan rasio H/W dalam koordinat pixel; menggunakan semua threshold upright yang dikonfigurasi; confidence parsing konsisten |
| `gap_v2` | Mempertahankan history untuk gap pendek yang kompatibel; real-frame warmup setelah diskontinuitas, bukan mengisi 30 salinan pose kembali; tidak mereset history hanya karena gerak turun cepat |
| `temporal_v2` | Tiga prediksi positif berurutan, recovery upright lima frame dengan skor rendah, dan gate gerakan kepala + pinggul berbasis waktu |

Gap pendek dibatasi **3 frame / 0.30 detik** dan memerlukan hip yang cukup
terlihat sebelum/sesudah gap serta pusat hip yang kompatibel. Frame pose hilang
tetap nol, sama seperti representasi training: **tidak ada interpolasi atau
keypoint buatan**. Tidak ada evaluasi neural/new alert pada frame tanpa pose;
reacquisition tetap perlu tiga observasi berurutan. Setelah gap panjang, target
berubah, resolusi berubah, atau stream terputus, history dibersihkan dan diisi
kembali dengan frame sungguhan. Alert yang telah terkonfirmasi tetap ditahan
melalui diskontinuitas visual sampai hold berakhir; target baru yang eksplisit
mereset evidence/latch detector (bukan latch ESTOP eksternal).

Pada `temporal_v2`, model positif masih diperlukan: motion tidak membuat alert
sendiri. Gerak turun kepala dan kedua hip yang cukup terlihat dapat melewati
veto upright; gerak bangkit memblokir kandidat baru. Vote dihitung dari skor
positif **sebelum** veto, sehingga tiga skor positif saat transisi dapat
mengonfirmasi ketika gate mengizinkan, bukan harus tiga frame non-upright.
Postur unknown bukan bukti upright dan masih dapat lolos konfirmasi model.

### Kalibrasi full-runtime (validation saja)

```powershell
python scripts/calibrate_runtime.py --output results/logs/runtime_v2/calibration.json
python scripts/compare_runtime_candidates.py --calibration results/logs/runtime_v2/calibration.json
```

Kalibrasi menjalankan ulang gate, reacquisition, voting, recovery dan latch
untuk setiap threshold **0.50–0.99** pada 50 video validation. Yang di-cache
hanya prediksi model dengan hash input identik; keputusan alert tidak di-cache.
Pemilihan mengutamakan recall **onset alert baru dalam interval Fall**,
lalu false alarm ADL/prematur, latency onset, threshold lebih tinggi dan profil
lebih sederhana sebagai tie-break. Alert palsu sebelum jatuh yang hanya
bertahan sampai interval Fall **tidak dihitung** sebagai onset benar. Alert
setelah interval dilaporkan terpisah, bukan memperpanjang anotasi untuk menaikkan
recall. Tidak ada penentuan parameter dari skor test.

Output kalibrasi tidak boleh ditimpa (gunakan nama output baru untuk eksperimen
berikutnya). File mencatat SHA-256 classifier, kode, extractor, fitur, sumber,
split validation, anotasi dan fixed settings. Perbandingan memakai pilihan yang
sudah dibekukan, menulis ringkasan `results/runtime_v2_comparison.json`, dan
memastikan bobot/scaler aktif tidak berubah. Detail tetap lokal/diabaikan Git.
Model kandidat dan cache juga lokal; menjalankan di komputer lain memerlukan
artefak tersebut atau ekstraksi/training/kalibrasi ulang.

### Uji robot yang disengaja (bukan aktivasi otomatis)

Hanya sesudah memeriksa hasil dan dengan pengawasan langsung, tanpa gerakan
robot yang berbahaya, opt-in memakai laporan kalibrasi yang diverifikasi:

```powershell
$env:HAFIBOT_FALL_RUNTIME_CALIBRATION = 'results/logs/runtime_v2/calibration.json'
python src/server_app.py --host 0.0.0.0 --port 5000
# Untuk kembali ke default, hentikan server lalu:
Remove-Item Env:HAFIBOT_FALL_RUNTIME_CALIBRATION
```

Server menolak opt-in jika hash classifier, extractor, kode atau settings tidak
cocok. Variabel tersebut **tidak diset otomatis**. Gunakan stream sekitar 15 FPS
seperti training; replay source-time tidak menjamin perilaku yang sama saat
frame jaringan terlambat/hilang. Input geometri menggunakan ukuran frame asli
yang diterima server.

### Hasil validation yang dibekukan (10 Oktober 2026)

Pemilihan full-runtime memilih **`temporal_v2` / threshold 0.72**: onset Fall
**18/19 (94.7%)**, false alarm **1/31 video ADL**, median delay onset **0.70 s**.
Threshold ini berlaku untuk classifier YOLO11m kandidat beserta profil tersebut,
**bukan** pengganti threshold model aktif atau angka universal.

| Profil | Threshold terbaik validation | Onset Fall | ADL false alarm | Median delay onset |
|---|---:|---:|---:|---:|
| `legacy_v1` | 0.86 | 18/19 | 1/31 | 1.10 s |
| `geometry_v2` | 0.78 | 18/19 | 1/31 | 0.90 s |
| `gap_v2` | 0.78 | 18/19 | 2/31 | 0.90 s |
| `temporal_v2` | 0.72 | 18/19 | 1/31 | 0.70 s |

Semua baris memakai classifier kandidat yang sama. Tidak semua perubahan
meningkatkan hasil: mempertahankan history juga dapat mempertahankan prediksi
palsu. Satu Fall validation (`old:Fall_ke_depan_8`) kehilangan pose pada
**59/76 frame** dan tidak menghasilkan evaluasi neural sama sekali pada replay;
menurunkan threshold tidak memperbaiki ketiadaan bukti visual. False alarm
validation berasal dari satu klip berbaring sengaja. Hanya ada **193 detik ADL**
validation; itu tidak cukup untuk menilai reliability monitoring seharian.

Replay regression pada 50 klip test yang **sudah pernah diperiksa**:

| Classifier / runtime | Threshold | Onset Fall dalam interval | ADL false alarm | Median delay |
|---|---:|---:|---:|---:|
| Aktif lama / legacy | 0.90 | 18/19 | 1/31 | 1.17 s |
| YOLO11m kandidat / legacy | 0.90 | 17/19 | 0/31 | 1.07 s |
| YOLO11m kandidat / geometry | 0.90 | 18/19 | 1/31 | 0.73 s |
| YOLO11m kandidat / gap | 0.90 | 18/19 | 1/31 | 0.73 s |
| YOLO11m kandidat / temporal | 0.90 | 18/19 | 0/31 | 0.80 s |
| YOLO11m kandidat / **validation-selected temporal** | **0.72** | **19/19** | **1/31** | **0.73 s** |

Profil gap/temporal pada 0.90 masing-masing punya satu alert terlambat setelah
interval Fall; itu tetap dihitung miss interval. Semua onset pada baris
validation-selected berada dalam interval dan tidak ada alert prematur.
**19/19 bukan bukti recall lapangan 100%**: hanya 19 jatuh, klip sudah pernah
digunakan untuk diagnosis, dan belum ada evaluasi orang/ruangan baru ataupun
robot bergerak. Dibanding kandidat legacy, recall bertambah dengan biaya satu
ADL false alarm. Parameter tidak diubah setelah melihat hasil regression.

Verifikasi kode: **136 tests passed**, tiga warning deprecation yang sudah ada.
Untuk Windows dengan akses temp terbatas, gunakan `--basetemp` baru di dalam
workspace serta `-p no:cacheprovider`. Fixture integrasi chatbot lama dapat
memajukan tanggal prayer reminders; backup/restore file reminder dengan aman
agar side effect test tidak mengubah data operasional.

### Batas bukti dan pengujian berikutnya

- Video test sebelumnya sudah diperiksa untuk diagnosis; hasil berikutnya adalah
  **regression**, bukan final unseen test. Split masih video-disjoint, bukan
  subject-disjoint. Butuh rekaman orang/ruangan baru yang tidak digunakan untuk
  pemilihan profil, threshold, atau diagnosis.
- Rasio/sudut image-plane bukan geometri 3D yang terkalibrasi. Perspektif kamera
  rendah, lensa wide-angle, dan resizing kamera tetap merupakan keterbatasan.
- Gerakan kamera robot dapat menyerupai gerak tubuh. Validasi stationary-video
  saja tidak cukup: uji robot diam/bergerak, duduk cepat, membungkuk, berbaring
  sengaja, keluar/masuk frame, occlusion dan pencahayaan berbeda.
- Hip-distance check bukan identitas orang. API kandidat menerima
  `target_changed=True`, tetapi YOLO saat ini belum memberikan person-ID yang
  andal ke server; batasi uji awal ke satu orang. Jangan menganggap kontinuitas
  multi-person telah diselesaikan.
- False alerts/hour dari klip ADL pendek diberi label **short-clip proxy**,
  bukan estimasi ADL kontinu. Rekam sesi ADL panjang, ukur alert episodes/jam,
  latency capture-to-alert, missed falls dan delivery notification/ESTOP.
- Tidak ada ground-truth jatuh saat pose di luar frame: missing pose bukan
  alasan memicu Fall. Candidate ini tidak mengganti uji keselamatan lapangan.
