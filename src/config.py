"""
config.py - Konfigurasi terpusat untuk seluruh pipeline Fall Detection.

Semua konstanta, path, dan hyperparameter didefinisikan di sini
agar konsisten di seluruh modul (extract, train, evaluate, inference).
"""

import os
from pathlib import Path

# ============================================================
# PATH
# ============================================================
# Root project (satu level di atas folder src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_YOLO_DIR = DATA_DIR / "processed_yolo"

# Subfolder per kelas
RAW_ADL_DIR = RAW_DIR / "ADL"
RAW_FALL_DIR = RAW_DIR / "Fall"
PROCESSED_ADL_DIR = PROCESSED_DIR / "ADL"
PROCESSED_FALL_DIR = PROCESSED_DIR / "Fall"
FALL_ANNOTATIONS_PATH = DATA_DIR / "fall_annotations.json"

# The July recordings were accidentally encoded at 15 FPS while containing
# roughly 30 captured frames per second, so they played at half speed. Keep the
# original paths explicit for audit/rollback, but train only from the repaired
# real-time copy (every second source frame, still encoded at 15 FPS). The July
# set stays in a separate processed directory because many filenames overlap
# the original dataset.
NEW_JULY_SLOWMO_RAW_DIR = DATA_DIR / "dataset new july"
NEW_JULY_SLOWMO_PROCESSED_DIR = DATA_DIR / "processed_new_july"
NEW_JULY_SLOWMO_ANNOTATIONS_PATH = (
    DATA_DIR / "fall_annotations_dataset_new_july.json"
)
NEW_JULY_RAW_DIR = DATA_DIR / "dataset new july realtime"
NEW_JULY_PROCESSED_DIR = DATA_DIR / "processed_new_july_realtime"
NEW_JULY_PROCESSED_YOLO_DIR = DATA_DIR / "processed_yolo_new_july_realtime"
NEW_JULY_ANNOTATIONS_PATH = (
    DATA_DIR / "fall_annotations_dataset_new_july_realtime.json"
)
NEW_AUG_RAW_DIR = DATA_DIR / "new dataset 20 aug"
NEW_AUG_PROCESSED_DIR = DATA_DIR / "processed_new_dataset_20_aug"
NEW_AUG_PROCESSED_YOLO_DIR = DATA_DIR / "processed_yolo_new_dataset_20_aug"
NEW_AUG_ANNOTATIONS_PATH = (
    DATA_DIR / "fall_annotations_new_dataset_20_aug.json"
)
SUBJECT_METADATA_PATH = DATA_DIR / "subject_metadata.json"

# Every source is loaded independently, then grouped by source video during
# train/validation/test splitting. Dataset prefixes prevent stem collisions.
TRAINING_DATASETS = (
    {
        "name": "old",
        "raw_root": RAW_DIR,
        "processed_adl_dir": PROCESSED_YOLO_DIR / "ADL",
        "processed_fall_dir": PROCESSED_YOLO_DIR / "Fall",
        "fall_annotations_path": FALL_ANNOTATIONS_PATH,
    },
    {
        "name": "new_july",
        "raw_root": NEW_JULY_RAW_DIR,
        "processed_adl_dir": NEW_JULY_PROCESSED_YOLO_DIR / "ADL",
        "processed_fall_dir": NEW_JULY_PROCESSED_YOLO_DIR / "Fall",
        "fall_annotations_path": NEW_JULY_ANNOTATIONS_PATH,
    },
    {
        "name": "new_aug",
        "raw_root": NEW_AUG_RAW_DIR,
        "processed_adl_dir": NEW_AUG_PROCESSED_YOLO_DIR / "ADL",
        "processed_fall_dir": NEW_AUG_PROCESSED_YOLO_DIR / "Fall",
        "fall_annotations_path": NEW_AUG_ANNOTATIONS_PATH,
    },
)

# Model & hasil
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# One pose configuration shared by offline extraction and live inference.
YOLO_POSE_MODEL_PATH = MODEL_DIR / "yolo11m-pose.pt"
YOLO_POSE_CONFIDENCE = 0.35
YOLO_POSE_INPUT_SIZE = (640, 640)
YOLO11_FEATURE_ROOT = DATA_DIR / "processed_yolo11m"


def training_dataset_sources(feature_root=None):
    """Select an isolated feature cache while preserving source IDs/annotations."""
    if feature_root is None:
        return TRAINING_DATASETS
    root = Path(feature_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return tuple({
        **source,
        "processed_adl_dir": root / source["name"] / "ADL",
        "processed_fall_dir": root / source["name"] / "Fall",
    } for source in TRAINING_DATASETS)

# ============================================================
# LABEL MAPPING
# ============================================================
LABEL_MAP = {
    "ADL": 0,    # Aktivitas normal (Activities of Daily Living)
    "Fall": 1,   # Jatuh
}
NUM_CLASSES = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())

# ============================================================
# MEDIAPIPE POSE (Tasks API - versi baru)
# ============================================================
# MediaPipe Pose menghasilkan 33 landmark, masing-masing punya (x, y, z, visibility)
MP_NUM_LANDMARKS = 33
MP_FEATURES_PER_LANDMARK = 4  # x, y, z, visibility
MP_TOTAL_FEATURES = MP_NUM_LANDMARKS * MP_FEATURES_PER_LANDMARK  # 132

# --- Landmark Selection ---
# Hanya 13 landmark kunci yang relevan untuk fall detection.
# Menghilangkan detail wajah (mata, telinga, mulut), jari tangan, dan jari kaki
# untuk mengurangi noise dan mempercepat inferensi.
SELECTED_LANDMARKS = [
    0,          # nose        — posisi kepala (turun drastis saat jatuh)
    11, 12,     # shoulders   — orientasi tubuh atas
    13, 14,     # elbows      — gerakan lengan
    15, 16,     # wrists      — refleks menahan badan saat jatuh
    23, 24,     # hips        — pusat massa tubuh (paling krusial!)
    25, 26,     # knees       — posisi kaki saat jatuh
    27, 28,     # ankles      — posisi kaki bawah
]
NUM_SELECTED_LANDMARKS = len(SELECTED_LANDMARKS)  # 13
SELECTED_FEATURES = NUM_SELECTED_LANDMARKS * MP_FEATURES_PER_LANDMARK  # 52

LEGACY_FEATURE_PIPELINE = "legacy_v1"
HYBRID_FEATURE_PIPELINE = "hybrid_body_v1"
# Conservative default retained after LOSO ablation: hybrid_body_v1 reduced
# held-out Fall recall. It remains available as an explicit experiment.
FEATURE_PIPELINE = LEGACY_FEATURE_PIPELINE
HYBRID_GLOBAL_FEATURE_NAMES = (
    "hip_center_y",
    "shoulder_center_y",
    "bbox_center_y",
    "bbox_width",
    "bbox_height",
    "bbox_aspect_ratio",
    "torso_scale",
)
HYBRID_GLOBAL_FEATURES = len(HYBRID_GLOBAL_FEATURE_NAMES)
BODY_RELATIVE_SCALE_MIN = 0.03
BODY_RELATIVE_COORD_CLIP = 10.0

# Gunakan tier Lite agar ekstraksi/training mendekati deployment Jetson yang
# menjalankan MediaPipe Legacy Pose dengan model_complexity=0 (Lite).
# Tasks API di desktop dan Solutions API di Jetson tidak sepenuhnya identik,
# tetapi tier model yang sama mengurangi domain shift dibanding Heavy -> Lite.
MP_MODEL_VARIANT = "lite"
MP_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"

# Semua video dataset saat ini direkam pada 15 FPS. Nilai ini disimpan sebagai
# metadata dan divalidasi saat ekstraksi supaya durasi window tidak berubah diam-diam.
EXPECTED_VIDEO_FPS = 15.0

# Confidence threshold untuk deteksi pose (Tasks API)
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ============================================================
# SEQUENCE / WINDOWING
# ============================================================
SEQUENCE_LENGTH = 30    # Jumlah frame per sequence
STRIDE = 5              # Langkah sliding window (83% overlap, lebih banyak data)
# Window Fall harus mengandung setidaknya 0.4 detik gerakan Fall pada 15 FPS.
# Window dengan 1..5 frame Fall dianggap boundary ambigu dan tidak dilatih.
FALL_MIN_OVERLAP_FRAMES = 6
# Active baseline threshold. Historical YOLOv8n-cache evaluation reported
# 98.0% video-max accuracy; this is not a live YOLO11m/posture-gated guarantee.
FALL_CONFIDENCE_THRESHOLD = 0.90

# ============================================================ 
# DATA SPLIT
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Video-level stratified split (default active model method)
SPLIT_STRATEGY = "video_stratified"
SUBJECT_TRAIN_IDS = ("S01", "S03")
SUBJECT_VAL_IDS = ("S04",)
SUBJECT_TEST_IDS = ("S02",)

# K-Fold Cross-Validation
N_FOLDS = 5             # Jumlah fold untuk K-Fold CV

# ============================================================
# FEATURE ENGINEERING
# ============================================================
USE_VELOCITY = True              # Tambahkan fitur kecepatan (delta antar frame)
USE_ACCELERATION = True          # Tambahkan fitur percepatan (delta dari velocity)
# Jumlah fitur velocity/acceleration: 13 selected landmarks × 3 (x,y,z only) = 39 per jenis
LANDMARK_DERIVATIVE_FEATURES = NUM_SELECTED_LANDMARKS * 3  # 39
BASE_FEATURES = SELECTED_FEATURES
VELOCITY_FEATURES = LANDMARK_DERIVATIVE_FEATURES
if FEATURE_PIPELINE == HYBRID_FEATURE_PIPELINE:
    BASE_FEATURES += HYBRID_GLOBAL_FEATURES
    VELOCITY_FEATURES += HYBRID_GLOBAL_FEATURES
# Total input features dihitung otomatis
INPUT_FEATURES = BASE_FEATURES
if USE_VELOCITY:
    INPUT_FEATURES += VELOCITY_FEATURES
if USE_ACCELERATION:
    INPUT_FEATURES += VELOCITY_FEATURES

# ============================================================
# DATA AUGMENTATION (hanya saat training)
# ============================================================
USE_AUGMENTATION = True
AUGMENTATION_PROFILE = "legacy_v1"
AUG_JITTER_STD = 0.005          # Noise xyz sebelum StandardScaler
AUG_SCALE_RANGE = (0.95, 1.05)  # Mild camera/body scale variation
AUG_TIME_WARP_PROB = 0.3        # Probabilitas time warping
AUG_DROPOUT_PROB = 0.0          # Legacy independent dropout disabled
AUG_HORIZONTAL_FLIP_PROB = 0.5
AUG_CAMERA_MOTION_PROB = 0.4
AUG_CAMERA_TRANSLATE = 0.03
AUG_CAMERA_ROTATE_DEG = 5.0
AUG_CONTIGUOUS_OCCLUSION_PROB = 0.3
AUG_CONTIGUOUS_OCCLUSION_MAX_FRAMES = 5

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 25     # Lebih sabar agar model konvergen
WEIGHT_DECAY = 5e-4              # Dikurangi agar tidak underfitting

# ============================================================
# MODEL ARCHITECTURE (diperbesar untuk dataset 193 video)
# ============================================================
# 1D CNN
CNN_FILTERS = [64, 128]          # Diperbesar untuk tangkap pola lebih kompleks
CNN_KERNEL_SIZE = 3              # Ukuran kernel
CNN_DROPOUT = 0.3                # Dikurangi agar tidak underfitting

# LSTM / GRU
RNN_TYPE = "LSTM"                # "LSTM" atau "GRU"
RNN_HIDDEN_SIZE = 128            # Diperbesar untuk kapasitas temporal
RNN_NUM_LAYERS = 2               # 2 layer LSTM
RNN_DROPOUT = 0.3                # Dropout antar LSTM layers
RNN_BIDIRECTIONAL = True         # Tetap bidirectional

# Attention
USE_ATTENTION = True             # Gunakan Attention mechanism

# Fully Connected
FC_HIDDEN_SIZE = 64              # Diperbesar
FC_DROPOUT = 0.4                 # Dikurangi sedikit

# ============================================================
# DEVICE
# ============================================================
DEVICE = "cuda"  # Akan di-override di runtime jika CUDA tidak tersedia

# ============================================================
# BUAT DIREKTORI OTOMATIS
# ============================================================
for dir_path in [PROCESSED_ADL_DIR, PROCESSED_FALL_DIR, MODEL_DIR, FIGURES_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
