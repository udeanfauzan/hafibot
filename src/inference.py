"""
inference.py - Deteksi jatuh real-time menggunakan webcam.

Pipeline real-time:
    Webcam → MediaPipe Pose → Feature Engineering → Sliding Window Buffer
    → CNN-LSTM Model → Visual Overlay + Alert

Penggunaan:
    python src/inference.py                  # Webcam default (index 0)
    python src/inference.py --source 1       # Webcam lain
    python src/inference.py --source video.mp4  # Dari file video

Kontrol keyboard:
    q / ESC : Keluar
    r       : Reset buffer
    s       : Screenshot
    m       : Toggle mute audio alert
"""

import sys
import time
import argparse
import numpy as np

# Polyfill untuk kompatibilitas model yang ditraining di NumPy 2.0 (PC) 
# lalu di-load di NumPy 1.19 (Jetson Nano)
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = sys.modules.get('numpy.core', None)
    sys.modules['numpy._core.multiarray'] = sys.modules.get('numpy.core.multiarray', None)

import cv2
import torch
import mediapipe as mp
from pathlib import Path
from collections import deque
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_DIR, FIGURES_DIR,
    SEQUENCE_LENGTH, INPUT_FEATURES,
    MP_NUM_LANDMARKS, MP_TOTAL_FEATURES,
    SELECTED_LANDMARKS, NUM_SELECTED_LANDMARKS, SELECTED_FEATURES,
    MP_MIN_DETECTION_CONFIDENCE, MP_MIN_POSE_PRESENCE_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE, MP_MODEL_PATH,
    USE_VELOCITY, USE_ACCELERATION, VELOCITY_FEATURES,
    CLASS_NAMES, DEVICE, FALL_CONFIDENCE_THRESHOLD,
    FEATURE_PIPELINE,
)
from src.model import (
    FallDetectionModel,
    checkpoint_feature_pipeline,
    checkpoint_input_features,
    checkpoint_rnn_type,
)
from src.dataset import engineer_selected_features
from src.posture import (
    analyze_posture,
    DEFAULT_MIN_UPRIGHT_TORSO_ANGLE,
    DEFAULT_MIN_UPRIGHT_ASPECT_RATIO,
    DEFAULT_MIN_DY_TORSO,
)
from src.yolo_detector import YOLOPoseDetector

# MediaPipe Tasks API imports
try:
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    MP_USE_TASKS = True
except AttributeError:
    # Jika gagal import, berarti menggunakan MediaPipe versi lama (misal 0.8.4 di Jetson Nano)
    MP_USE_TASKS = False

# Coba import winsound untuk alert audio (Windows only)
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ============================================================
# KONSTANTA VISUAL
# ============================================================
# Warna (BGR format untuk OpenCV)
COLOR_GREEN = (0, 220, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 160, 0)
COLOR_ORANGE = (0, 140, 255)
COLOR_SKELETON = (255, 200, 50)
COLOR_JOINT = (0, 255, 200)
COLOR_PANEL_BG = (30, 30, 30)

# MediaPipe Pose connections (pasangan landmark untuk menggambar skeleton)
POSE_CONNECTIONS = [
    # Wajah
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Badan atas
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    # Tangan
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    # Badan bawah
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

JOINTS_13_CONNECTIONS = [
    (0, 1), (0, 2), (1, 2),  # Head to shoulders
    (1, 3), (3, 5),          # Left arm
    (2, 4), (4, 6),          # Right arm
    (1, 7), (2, 8), (7, 8),  # Torso
    (7, 9), (9, 11),         # Left leg
    (8, 10), (10, 12),       # Right leg
]

# Pengaturan alert
ALERT_COOLDOWN_SEC = 3.0      # Cooldown antar alert (detik)
ALERT_FLASH_DURATION = 0.5    # Durasi flash merah (detik)
CONFIDENCE_THRESHOLD = FALL_CONFIDENCE_THRESHOLD
POSE_LOSS_GRACE_SEC = 2.0     # Batas sebelum buffer pose lama diprime ulang
FALL_HOLD_SEC = 5.0           # Pertahankan status fall agar event tidak terlewat
POSE_REACQUIRE_CONFIRMATION_FRAMES = 3  # Stabilkan pose sebelum fall baru
BEEP_FREQ = 1500              # Frekuensi beep (Hz)
BEEP_DURATION_MS = 300        # Durasi beep (ms)


# ============================================================
# FALL DETECTOR CLASS
# ============================================================
class FallDetector:
    """
    Detektor jatuh real-time menggunakan webcam.

    Alur kerja:
        1. Capture frame dari webcam
        2. Deteksi pose dengan MediaPipe → 132 fitur/frame
        3. Simpan di circular buffer (30 frame terakhir)
        4. Saat buffer penuh: feature engineering → normalize → predict
        5. Tampilkan overlay visual + alert jika jatuh terdeteksi
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 pose_loss_grace_sec: float = POSE_LOSS_GRACE_SEC,
                 fall_hold_sec: float = FALL_HOLD_SEC,
                 pose_reacquire_confirmation_frames: int =
                 POSE_REACQUIRE_CONFIRMATION_FRAMES,
                 enable_posture_filter: bool = True,
                 min_upright_torso_angle: float = DEFAULT_MIN_UPRIGHT_TORSO_ANGLE,
                 min_upright_aspect_ratio: float = DEFAULT_MIN_UPRIGHT_ASPECT_RATIO,
                 min_dy_torso: float = DEFAULT_MIN_DY_TORSO):
        """
        Inisialisasi FallDetector.

        Args:
            confidence_threshold: Threshold probabilitas untuk klasifikasi Fall
            pose_loss_grace_sec: Setelah pose hilang lebih lama dari durasi ini,
                buffer diprime ulang secara instan dari pose pertama yang kembali
                agar frame nol lama tidak memicu false fall.
            fall_hold_sec: Durasi minimum status fall dipertahankan setelah
                terkonfirmasi, termasuk ketika MediaPipe kehilangan pose.
            pose_reacquire_confirmation_frames: Jumlah pose valid berturut-turut
                sebelum prediksi model boleh memulai event fall baru.
            enable_posture_filter: Aktifkan verifikasi postur geometris (posture gating)
                untuk memblokir false alarm saat pengguna bergerak/menjauh berdiri tegak.
            min_upright_torso_angle: Sudut minimum kemiringan torso (derajat) untuk dianggap tegak.
            min_upright_aspect_ratio: Rasio aspek minimum (H/W) untuk dianggap tegak.
            min_dy_torso: Jarak vertikal minimum bahu di atas pinggul.
        """
        self.confidence_threshold = confidence_threshold
        self.pose_loss_grace_sec = max(0.0, float(pose_loss_grace_sec))
        self.fall_hold_sec = max(0.0, float(fall_hold_sec))
        self.pose_reacquire_confirmation_frames = max(
            1, int(pose_reacquire_confirmation_frames)
        )
        self.enable_posture_filter = bool(enable_posture_filter)
        self.min_upright_torso_angle = float(min_upright_torso_angle)
        self.min_upright_aspect_ratio = float(min_upright_aspect_ratio)
        self.min_dy_torso = float(min_dy_torso)
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

        # State
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)  # Circular buffer 30 frame
        self.current_prediction = "Menunggu..."
        self.current_confidence = 0.0
        self.current_landmarks = None
        self.is_fall = False
        self.mute = False
        self.pose_missing_frames = 0
        self.pose_detected_streak = 0
        self.upright_streak = 0
        self.last_hip_center = None
        self.trajectory_history = deque(maxlen=30)
        self.last_pose_monotonic = None
        self.fall_hold_until = 0.0

        # Timing
        self.last_alert_time = 0.0
        self.flash_start_time = 0.0
        self.fps_history = deque(maxlen=30)
        self.frame_count = 0

        # Load model & scaler
        self._load_model()
        self._load_scaler()

        print(f"  FallDetector siap!")
        print(f"  Device       : {self.device}")
        print(f"  Buffer size  : {SEQUENCE_LENGTH} frames")
        print(f"  Threshold    : {self.confidence_threshold:.0%}")
        print(f"  Pose grace   : {self.pose_loss_grace_sec:.1f} detik")
        print(
            f"  Pose confirm : {self.pose_reacquire_confirmation_frames} frames"
        )
        print(f"  Fall hold    : {self.fall_hold_sec:.1f} detik")
        print(f"  Posture gate : {'ON' if self.enable_posture_filter else 'OFF'} (Angle>={self.min_upright_torso_angle:.0f} deg, AR>={self.min_upright_aspect_ratio:.2f})")
        print(f"  Audio alert  : {'ON' if HAS_WINSOUND else 'N/A (bukan Windows)'}")

    def _load_model(self):
        """Load model CNN-LSTM dari checkpoint."""
        model_path = MODEL_DIR / "best_model.pt"
        if not model_path.exists():
            print(f"  [ERROR] Model tidak ditemukan: {model_path}")
            print("  Jalankan: python src/train.py")
            sys.exit(1)

        try:
            checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(model_path), map_location=self.device)
        self.checkpoint = checkpoint
        self.feature_pipeline = checkpoint_feature_pipeline(checkpoint)
        self.input_features = checkpoint_input_features(checkpoint)
        self.rnn_type = checkpoint_rnn_type(checkpoint)
        self.model = FallDetectionModel(
            input_size=self.input_features, rnn_type=self.rnn_type
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        val_acc = checkpoint.get("val_acc", 0)
        checkpoint_stage = (
            f"Epoch {checkpoint['epoch']}"
            if "epoch" in checkpoint else f"Fold {checkpoint.get('fold', '?')}"
        )
        print(f"  Model loaded : {checkpoint_stage}, Val Acc={val_acc:.1%}")

    def _load_scaler(self):
        """Load scaler yang terikat dengan checkpoint, dengan fallback legacy."""
        expected_features = getattr(self, "input_features", INPUT_FEATURES)
        scaler_path = MODEL_DIR / "scaler_params.npz"
        if "scaler_mean" in self.checkpoint and "scaler_std" in self.checkpoint:
            self.scaler_mean = np.asarray(
                self.checkpoint["scaler_mean"], dtype=np.float32
            )
            self.scaler_std = np.asarray(
                self.checkpoint["scaler_std"], dtype=np.float32
            )
            scaler_source = "checkpoint"
        else:
            if not scaler_path.exists():
                print(f"  [ERROR] Scaler tidak ditemukan: {scaler_path}")
                sys.exit(1)
            scaler = np.load(str(scaler_path))
            self.scaler_mean = scaler["mean"].astype(np.float32)
            self.scaler_std = scaler["std"].astype(np.float32)
            scaler_source = scaler_path.name

        if self.scaler_mean.shape != (expected_features,):
            raise ValueError(
                f"Scaler mean shape {self.scaler_mean.shape} tidak cocok "
                f"dengan checkpoint input_features={expected_features}"
            )
        if self.scaler_std.shape != (expected_features,):
            raise ValueError(
                f"Scaler std shape {self.scaler_std.shape} tidak cocok "
                f"dengan checkpoint input_features={expected_features}"
            )
        if not np.isfinite(self.scaler_mean).all() or not np.isfinite(self.scaler_std).all():
            raise ValueError("Scaler mengandung nilai non-finite")
        if np.any(self.scaler_std <= 0):
            raise ValueError("Scaler std harus lebih besar dari nol")

        print(
            f"  Scaler loaded: {len(self.scaler_mean)} features "
            f"(source={scaler_source})"
        )

    def _create_landmarker(self):
        yolo_pt = MODEL_DIR / "yolov8n-pose.pt"
        yolo_engine = MODEL_DIR / "yolov8n-pose.engine"
        yolo_onnx = MODEL_DIR / "yolov8n-pose.onnx"
        if yolo_engine.exists() or yolo_pt.exists() or yolo_onnx.exists():
            model_file = str(yolo_engine if yolo_engine.exists() else (yolo_pt if yolo_pt.exists() else yolo_onnx))
            return YOLOPoseDetector(model_path=model_file)
        elif MP_USE_TASKS:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=MP_MIN_POSE_PRESENCE_CONFIDENCE,
                min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
            )
            return PoseLandmarker.create_from_options(options)
        else:
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
            )

    def _extract_frame_features(self, landmarks) -> np.ndarray:
        if isinstance(landmarks, np.ndarray):
            if landmarks.size == SELECTED_FEATURES:
                return landmarks.flatten().astype(np.float32)
            elif landmarks.size == MP_TOTAL_FEATURES:
                return filter_landmarks(landmarks.reshape(1, -1))[0]

        if len(landmarks) == NUM_SELECTED_LANDMARKS:
            features = []
            for lm in landmarks:
                features.extend([float(lm.x), float(lm.y), float(getattr(lm, 'z', 0.0)), float(getattr(lm, 'visibility', 1.0))])
            return np.array(features, dtype=np.float32)

        features = []
        for idx in SELECTED_LANDMARKS:
            lm = landmarks[idx]
            features.extend([float(lm.x), float(lm.y), float(getattr(lm, 'z', 0.0)), float(getattr(lm, 'visibility', 1.0))])
        return np.array(features, dtype=np.float32)

    def _engineer_features(self, raw_buffer: np.ndarray) -> np.ndarray:
        """
        Apply feature engineering (velocity + acceleration) pada buffer.

        Args:
            raw_buffer: shape (SEQUENCE_LENGTH, 52)  — 13 selected landmarks

        Returns:
            np.ndarray shape (SEQUENCE_LENGTH, 130)
        """
        feature_pipeline = getattr(self, "feature_pipeline", FEATURE_PIPELINE)
        return engineer_selected_features(raw_buffer, feature_pipeline)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data menggunakan scaler dari training.

        Args:
            data: shape (SEQUENCE_LENGTH, INPUT_FEATURES)

        Returns:
            Normalized data, shape sama
        """
        return (data - self.scaler_mean) / self.scaler_std

    @torch.no_grad()
    def _predict(self, sequence: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Jalankan model inference pada satu sequence.

        Args:
            sequence: shape (SEQUENCE_LENGTH, INPUT_FEATURES), sudah normalized

        Returns:
            pred_class: 0 (ADL) atau 1 (Fall)
            confidence: probabilitas kelas terprediksi
            probs: array probabilitas semua kelas
        """
        # Buat tensor: (1, seq_len, features)
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        logits, attention_weights = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return pred_class, confidence, probs

    def process_frame(self, landmarks) -> dict:
        """
        Proses satu frame: update buffer, predict jika buffer penuh.

        Args:
            landmarks: MediaPipe pose landmarks (atau None jika tidak terdeteksi)

        Returns:
            dict dengan keys: prediction, confidence, is_fall, buffer_status, fall_prob
        """
        now = time.monotonic()
        person_detected = landmarks is not None
        was_fall = self.is_fall
        missing_frames_before_current = self.pose_missing_frames
        pose_loss_age_before_current = (
            now - self.last_pose_monotonic
            if self.last_pose_monotonic is not None else float("inf")
        )
        buffer_reprimed = False

        if person_detected:
            features = self._extract_frame_features(landmarks)
            self.pose_detected_streak += 1
            current_hip = ((features[28] + features[32]) * 0.5, (features[29] + features[33]) * 0.5)

            # REPRIME BUFFER ON POSE RETURN OR SPATIAL TARGET JUMP:
            # 1. On return after any missing frames, reprime buffer to eliminate (0 -> 1) velocity spikes.
            # 2. On spatial discontinuity (target switch/teleport > 18% of frame), reprime to erase jump spikes.
            target_jumped = False
            if self.last_hip_center is not None:
                dx = abs(current_hip[0] - self.last_hip_center[0])
                dy = abs(current_hip[1] - self.last_hip_center[1])
                if dx > 0.18 or dy > 0.18:
                    target_jumped = True
            self.last_hip_center = current_hip

            if missing_frames_before_current > 0 or target_jumped:
                self.frame_buffer.clear()
                self.frame_buffer.extend(
                    features.copy() for _ in range(SEQUENCE_LENGTH)
                )
                self.trajectory_history.clear()
                buffer_reprimed = True

            self.pose_missing_frames = 0
            self.last_pose_monotonic = now
        else:
            # Samakan perilaku runtime dengan preprocessing training: pose yang
            # gagal dideteksi direpresentasikan sebagai satu frame berisi nol.
            # CNN-LSTM sengaja tidak dijalankan pada frame ini: tanpa pose tidak
            # ada bukti visual baru, dan probabilitas Fall harus tetap netral.
            self.last_hip_center = None
            features = np.zeros(SELECTED_FEATURES, dtype=np.float32)
            self.pose_missing_frames += 1
            self.pose_detected_streak = 0

        if not buffer_reprimed:
            self.frame_buffer.append(features)
        buffer_len = len(self.frame_buffer)

        pose_loss_age = (
            now - self.last_pose_monotonic
            if self.last_pose_monotonic is not None else float("inf")
        )
        pose_grace_active = (
            person_detected or pose_loss_age <= self.pose_loss_grace_sec
        )

        # Analisis postur geometris
        posture_info = None
        posture_gated = False
        is_upright = False
        if person_detected:
            posture_info = analyze_posture(
                landmarks,
                min_upright_torso_angle=self.min_upright_torso_angle,
                min_upright_aspect_ratio=self.min_upright_aspect_ratio,
                min_dy_torso=self.min_dy_torso,
            )
            is_upright = bool(posture_info and posture_info.get('is_upright', False))
            if is_upright:
                self.upright_streak += 1
            else:
                self.upright_streak = 0

            # Track trajectory vertikal (hip_y dan nose_y)
            hip_y = (features[29] + features[33]) * 0.5
            nose_y = features[1]
            self.trajectory_history.append((hip_y, nose_y))
        else:
            self.upright_streak = 0
            self.trajectory_history.append((None, None))

        # Cek apakah arah pergerakan tubuh sedang naik ke atas (bangkit dari lantai)
        is_rising = False
        if len(self.trajectory_history) >= 15:
            valid_hips = [h[0] for h in list(self.trajectory_history)[-15:] if h[0] is not None]
            if len(valid_hips) >= 8:
                delta_hip = valid_hips[-1] - valid_hips[0]  # negatif = bergerak ke atas frame
                if delta_hip < -0.05:
                    is_rising = True

        model_prediction = self.current_prediction
        model_confidence = self.current_confidence
        fall_prob = 0.0
        model_is_fall = False

        # Prediksi hanya jika buffer penuh dan pose terlihat. Ini juga mengurangi
        # pekerjaan model saat scene kosong sehingga thread stream tidak terbebani.
        if person_detected and buffer_len >= SEQUENCE_LENGTH:
            raw_buffer = np.array(list(self.frame_buffer), dtype=np.float32)

            # Feature engineering
            engineered = self._engineer_features(raw_buffer)

            # Normalize
            normalized = self._normalize(engineered)

            # Predict
            pred_class, confidence, probs = self._predict(normalized)

            model_prediction = CLASS_NAMES[pred_class]
            model_confidence = confidence
            fall_prob = float(probs[1])
            model_is_fall = bool(
                pred_class == 1 and fall_prob >= self.confidence_threshold
            )

            # Posture Gating: Jika model memprediksi Fall tetapi tubuh terverifikasi
            # berdiri/berjalan tegak (upright), batalkan alert untuk mencegah false alarm
            # saat bergerak menjauh dari robot.
            if model_is_fall and self.enable_posture_filter:
                if is_upright:
                    model_is_fall = False
                    posture_gated = True
                    model_prediction = 'ADL (Postur Tegak)'
                    model_confidence = float(probs[0])
                    fall_prob = 0.0
                elif is_rising:
                    model_is_fall = False
                    posture_gated = True
                    model_prediction = 'ADL (Bangkit)'
                    model_confidence = float(probs[0])
                    fall_prob = 0.0

        # Pose yang hilang tidak boleh memulai event baru. Setelah pose kembali,
        # tunggu beberapa frame valid agar satu landmark reacquisition yang tidak
        # stabil tidak langsung mengubah stale model output menjadi alert.
        fall_eligible = bool(
            person_detected
            and buffer_len >= SEQUENCE_LENGTH
            and self.pose_detected_streak >= self.pose_reacquire_confirmation_frames
        )
        confirmed_fall = model_is_fall and fall_eligible
        if confirmed_fall:
            self.fall_hold_until = max(
                self.fall_hold_until, now + self.fall_hold_sec
            )

        fall_latched = now < self.fall_hold_until

        # Instant Dynamic De-Latch:
        # If person is confirmed upright (standing, walking, sitting) for >= 2 frames,
        # IMMEDIATELY CANCEL any latched fall alert! An upright person is NOT fallen.
        if self.enable_posture_filter and is_upright and self.upright_streak >= 1:
            self.fall_hold_until = 0.0
            fall_latched = False
            confirmed_fall = False

        self.is_fall = bool(confirmed_fall or fall_latched)

        if self.is_fall and not was_fall:
            self._trigger_alert()

        if person_detected:
            if buffer_len >= SEQUENCE_LENGTH and not fall_eligible:
                self.current_prediction = "Menstabilkan pose..."
                self.current_confidence = 0.0
            elif is_upright and posture_gated:
                self.current_prediction = 'ADL (Postur Tegak)'
                self.current_confidence = model_confidence
            else:
                self.current_prediction = model_prediction
                self.current_confidence = model_confidence
        elif self.is_fall:
            self.current_prediction = "Fall (pose hilang)"
            self.current_confidence = 0.0
        else:
            self.current_prediction = "Tidak Terdeteksi"
            self.current_confidence = 0.0

        result = {
            "prediction": self.current_prediction,
            "confidence": self.current_confidence,
            "is_fall": self.is_fall,
            "buffer_filled": buffer_len,
            "buffer_total": SEQUENCE_LENGTH,
            "fall_prob": fall_prob if fall_eligible else 0.0,
            "model_is_fall": model_is_fall,
            "model_evaluated": bool(person_detected and buffer_len >= SEQUENCE_LENGTH),
            "fall_eligible": fall_eligible,
            "pose_detected": person_detected,
            "pose_missing_frames": self.pose_missing_frames,
            "pose_detected_streak": self.pose_detected_streak,
            "pose_grace_active": pose_grace_active,
            "buffer_reprimed": buffer_reprimed,
            "fall_latched": bool(self.is_fall and not confirmed_fall),
            "posture": posture_info,
            "posture_gated": posture_gated,
        }

        return result

    def _trigger_alert(self):
        """Trigger alert visual + audio jika cooldown sudah lewat."""
        current_time = time.time()
        if current_time - self.last_alert_time >= ALERT_COOLDOWN_SEC:
            self.last_alert_time = current_time
            self.flash_start_time = current_time

            # Audio alert (non-blocking di thread terpisah)
            if HAS_WINSOUND and not self.mute:
                import threading
                threading.Thread(
                    target=winsound.Beep,
                    args=(BEEP_FREQ, BEEP_DURATION_MS),
                    daemon=True,
                ).start()

    def reset_buffer(self):
        """Reset buffer dan prediksi."""
        self.frame_buffer.clear()
        self.trajectory_history.clear()
        self.current_prediction = "Menunggu..."
        self.current_confidence = 0.0
        self.is_fall = False
        self.pose_missing_frames = 0
        self.pose_detected_streak = 0
        self.upright_streak = 0
        self.last_hip_center = None
        self.last_pose_monotonic = None
        self.fall_hold_until = 0.0
        print("  [INFO] Buffer direset.")

    # ========================================================
    # VISUAL OVERLAY
    # ========================================================
    def draw_overlay(self, frame: np.ndarray, result: dict,
                     landmarks=None, fps: float = 0.0) -> np.ndarray:
        """
        Gambar overlay visual pada frame.

        Args:
            frame: Frame BGR dari webcam
            result: Dict hasil dari process_frame()
            landmarks: MediaPipe landmarks untuk menggambar skeleton
            fps: FPS saat ini

        Returns:
            Frame dengan overlay
        """
        h, w = frame.shape[:2]

        # === Flash effect saat fall ===
        if result.get("is_fall", False):
            elapsed_flash = time.time() - self.flash_start_time
            if elapsed_flash < ALERT_FLASH_DURATION:
                # Overlay merah semi-transparan yang berkedip
                alpha = 0.3 * (1 - elapsed_flash / ALERT_FLASH_DURATION)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), COLOR_RED, -1)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # === Gambar skeleton ===
        if landmarks is not None:
            self._draw_skeleton(frame, landmarks, w, h)

        # === Panel info (kiri atas) ===
        self._draw_info_panel(frame, result, fps, w, h)

        # === Confidence bar (bawah) ===
        self._draw_confidence_bar(frame, result, w, h)

        # === Status besar (tengah atas) ===
        self._draw_status_label(frame, result, w, h)

        # === Buffer progress ===
        self._draw_buffer_progress(frame, result, w, h)

        return frame

    def _draw_skeleton(self, frame, landmarks, w, h):
        """Gambar skeleton pose pada frame."""
        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))
            vis = lm.visibility

            # Warna joint berdasarkan visibility
            if vis > 0.7:
                color = COLOR_JOINT
                radius = 4
            elif vis > 0.3:
                color = COLOR_YELLOW
                radius = 3
            else:
                color = (100, 100, 100)
                radius = 2

            cv2.circle(frame, (px, py), radius, color, -1)
            cv2.circle(frame, (px, py), radius + 1, COLOR_BLACK, 1)

        # Gambar connections
        connections = JOINTS_13_CONNECTIONS if len(points) == NUM_SELECTED_LANDMARKS else POSE_CONNECTIONS
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start_vis = landmarks[start_idx].visibility
                end_vis = landmarks[end_idx].visibility

                if start_vis > 0.3 and end_vis > 0.3:
                    color = COLOR_SKELETON
                    cv2.line(frame, points[start_idx], points[end_idx], color, 2, cv2.LINE_AA)

    def _draw_info_panel(self, frame, result, fps, w, h):
        """Gambar panel informasi di kiri atas."""
        panel_w = 290
        panel_h = 168
        margin = 10

        # Background panel semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (margin, margin),
                      (margin + panel_w, margin + panel_h),
                      COLOR_PANEL_BG, -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Border
        cv2.rectangle(frame, (margin, margin),
                      (margin + panel_w, margin + panel_h),
                      COLOR_BLUE, 1)

        # Teks info
        x_text = margin + 12
        y_start = margin + 28
        line_h = 26

        # FPS
        fps_color = COLOR_GREEN if fps >= 15 else COLOR_YELLOW if fps >= 10 else COLOR_RED
        cv2.putText(frame, f"FPS: {fps:.1f}", (x_text, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2, cv2.LINE_AA)

        # Buffer status
        buf_filled = result.get("buffer_filled", 0)
        buf_total = result.get("buffer_total", SEQUENCE_LENGTH)
        buf_pct = buf_filled / buf_total * 100
        buf_color = COLOR_GREEN if buf_filled >= buf_total else COLOR_YELLOW
        cv2.putText(frame, f"Buffer: {buf_filled}/{buf_total} ({buf_pct:.0f}%)",
                    (x_text, y_start + line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, buf_color, 1, cv2.LINE_AA)

        # Prediksi
        pred_text = result.get("prediction", "Menunggu...")
        conf = result.get("confidence", 0)
        pred_color = COLOR_RED if result.get("is_fall") else COLOR_GREEN
        cv2.putText(frame, f"Prediksi: {pred_text}", (x_text, y_start + line_h * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2, cv2.LINE_AA)

        # Confidence
        cv2.putText(frame, f"Confidence: {conf:.1%}", (x_text, y_start + line_h * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

        # Postur geometris
        posture = result.get("posture")
        if posture and posture.get("valid"):
            label = posture.get("posture_label", "")
            angle = posture.get("torso_angle_deg", 0.0)
            if result.get("posture_gated"):
                posture_text = f"Postur: {label.capitalize()} ({angle:.0f} deg) [GATED]"
                posture_color = COLOR_YELLOW
            else:
                posture_text = f"Postur: {label.capitalize()} ({angle:.0f} deg)"
                posture_color = COLOR_GREEN if posture.get("is_upright") else COLOR_ORANGE if posture.get("is_lying") else COLOR_WHITE
            cv2.putText(frame, posture_text, (x_text, y_start + line_h * 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, posture_color, 1, cv2.LINE_AA)

        # Mute indicator
        if self.mute:
            cv2.putText(frame, "[MUTED]", (margin + panel_w - 80, y_start),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ORANGE, 1, cv2.LINE_AA)

    def _draw_confidence_bar(self, frame, result, w, h):
        """Gambar confidence bar di bagian bawah frame."""
        fall_prob = result.get("fall_prob", 0.0)

        bar_h = 30
        bar_margin = 10
        bar_y = h - bar_h - bar_margin
        bar_x = bar_margin
        bar_w = w - 2 * bar_margin

        # Background bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      COLOR_PANEL_BG, -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), 1)

        # Fill bar
        fill_w = int(bar_w * fall_prob)
        if fall_prob >= self.confidence_threshold:
            bar_color = COLOR_RED
        elif fall_prob >= self.confidence_threshold * 0.7:
            bar_color = COLOR_ORANGE
        else:
            bar_color = COLOR_GREEN

        if fill_w > 0:
            cv2.rectangle(frame, (bar_x + 1, bar_y + 1),
                          (bar_x + fill_w, bar_y + bar_h - 1),
                          bar_color, -1)

        # Threshold line
        threshold_x = bar_x + int(bar_w * self.confidence_threshold)
        cv2.line(frame, (threshold_x, bar_y), (threshold_x, bar_y + bar_h),
                 COLOR_WHITE, 2)

        # Label
        cv2.putText(frame, f"Fall Prob: {fall_prob:.1%}",
                    (bar_x + 5, bar_y + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

        # ADL dan Fall labels
        cv2.putText(frame, "ADL", (bar_x + 5, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, "FALL", (bar_x + bar_w - 45, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_RED, 1, cv2.LINE_AA)

    def _draw_status_label(self, frame, result, w, h):
        """Gambar label status besar di tengah atas."""
        if result.get("buffer_filled", 0) < result.get("buffer_total", SEQUENCE_LENGTH):
            text = "MEMUAT BUFFER..."
            color = COLOR_YELLOW
            bg_color = (0, 80, 100)
        elif result.get("is_fall", False):
            text = "!! FALL DETECTED !!"
            color = COLOR_WHITE
            bg_color = (0, 0, 200)
        else:
            text = "NORMAL"
            color = COLOR_WHITE
            bg_color = (0, 120, 0)

        # Ukuran teks
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Posisi (tengah atas)
        x = (w - text_w) // 2
        y = 50

        # Background rounded rect
        pad = 12
        cv2.rectangle(frame, (x - pad, y - text_h - pad),
                      (x + text_w + pad, y + baseline + pad),
                      bg_color, -1)
        cv2.rectangle(frame, (x - pad, y - text_h - pad),
                      (x + text_w + pad, y + baseline + pad),
                      COLOR_WHITE, 2)

        # Teks
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def _draw_buffer_progress(self, frame, result, w, h):
        """Gambar progress bar buffer di atas confidence bar."""
        buf_filled = result.get("buffer_filled", 0)
        buf_total = result.get("buffer_total", SEQUENCE_LENGTH)

        if buf_filled >= buf_total:
            return  # Jangan tampilkan jika sudah penuh

        bar_h = 6
        bar_y = h - 50
        bar_margin = 10
        bar_x = bar_margin
        bar_w = w - 2 * bar_margin

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)

        # Fill
        fill_w = int(bar_w * buf_filled / buf_total)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      COLOR_BLUE, -1)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    def run(self, source=0):
        """
        Main loop: capture webcam, proses, dan tampilkan hasil.

        Args:
            source: 0 untuk webcam default, atau path ke file video
        """
        print(f"\n{'='*60}")
        print(f"  FALL DETECTION - Real-Time Inference")
        print(f"{'='*60}")
        print(f"  Source     : {'Webcam ' + str(source) if isinstance(source, int) else source}")
        print(f"  Threshold  : {self.confidence_threshold:.0%}")
        print(f"  Kontrol    : q=Keluar  r=Reset  s=Screenshot  m=Mute")
        print(f"{'='*60}\n")

        # Buka webcam/video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  [ERROR] Tidak bisa membuka source: {source}")
            sys.exit(1)

        # Set resolusi webcam
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"  Resolusi   : {actual_w}x{actual_h}")
        print(f"  Source FPS : {source_fps:.1f}")
        print(f"  Tekan 'q' untuk keluar.\n")

        # Buat window
        window_name = "Fall Detection - Real-Time"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        screenshot_count = 0

        # Buat MediaPipe PoseLandmarker
        with self._create_landmarker() as landmarker:
            while True:
                t_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, int):
                        print("  [ERROR] Webcam terputus!")
                        break
                    else:
                        print("  [INFO] Video selesai.")
                        break

                self.frame_count += 1

                # === MediaPipe Pose Detection ===
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Timestamp dalam milliseconds
                timestamp_ms = int(self.frame_count * (1000 / source_fps))

                result_mp = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Ambil landmarks (jika terdeteksi)
                landmarks = None
                if result_mp.pose_landmarks and len(result_mp.pose_landmarks) > 0:
                    landmarks = result_mp.pose_landmarks[0]

                # === Proses frame ===
                result = self.process_frame(landmarks)

                # === Hitung FPS ===
                t_end = time.time()
                frame_time = t_end - t_start
                if frame_time > 0:
                    self.fps_history.append(1.0 / frame_time)
                fps = np.mean(self.fps_history) if self.fps_history else 0

                # === Gambar overlay ===
                frame = self.draw_overlay(frame, result, landmarks, fps)

                # === Tampilkan ===
                cv2.imshow(window_name, frame)

                # === Keyboard handling ===
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q atau ESC
                    print("  [INFO] Keluar...")
                    break

                elif key == ord('r'):
                    self.reset_buffer()

                elif key == ord('s'):
                    screenshot_count += 1
                    ss_path = FIGURES_DIR / f"screenshot_{screenshot_count:03d}.png"
                    cv2.imwrite(str(ss_path), frame)
                    print(f"  [INFO] Screenshot disimpan: {ss_path}")

                elif key == ord('m'):
                    self.mute = not self.mute
                    status = "MUTED" if self.mute else "UNMUTED"
                    print(f"  [INFO] Audio alert: {status}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n  Selesai! Total frame diproses: {self.frame_count}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection - Real-Time Inference via Webcam"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Sumber video: 0 untuk webcam (default), atau path ke file video"
    )
    parser.add_argument(
        "--threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Threshold probabilitas untuk deteksi fall (default: {CONFIDENCE_THRESHOLD})"
    )
    args = parser.parse_args()

    # Parse source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Validasi file jika bukan webcam
    if isinstance(source, str) and not Path(source).exists():
        print(f"  [ERROR] File video tidak ditemukan: {source}")
        sys.exit(1)

    detector = FallDetector(confidence_threshold=args.threshold)
    detector.run(source=source)


if __name__ == "__main__":
    main()
