'''
yolo_detector.py - Unified YOLOv8-Pose Detector for PC & Jetson Nano TensorRT.

Supports:
1. PyTorch (.pt) via Ultralytics (PC / CUDA / CPU)
2. ONNX (.onnx) via ONNXRuntime (PC / Jetson GPU/CPU)
3. TensorRT (.engine) via TensorRT Python API (Jetson Nano GPU FP16)

Outputs standardized 13 selected joints matching the fall detection pipeline:
[nose, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist,
 left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]
'''

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np
import torch

from src.config import (
    YOLO_POSE_MODEL_PATH, YOLO_POSE_CONFIDENCE, YOLO_POSE_INPUT_SIZE,
)

# COCO 17 Keypoints -> 13 Selected Keypoints mapping
# 0: nose, 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist,
# 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle
COCO_SELECTED_KEYPOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_SELECTED_JOINTS = len(COCO_SELECTED_KEYPOINTS)  # 13


class PoseKeypoint:
    '''Standardized single keypoint compatible with MediaPipe landmark interface.'''
    __slots__ = ('x', 'y', 'z', 'visibility', 'presence')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, visibility: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(visibility)

    def __repr__(self):
        return f'PoseKeypoint(x={self.x:.3f}, y={self.y:.3f}, v={self.visibility:.2f})'


class YOLOLandmarksWrapper:
    '''Wrapper to mimic MediaPipe pose_landmarks attribute and indexing.'''
    def __init__(self, landmarks: Optional[List[PoseKeypoint]]):
        self.landmark = landmarks if landmarks is not None else []

    def __iter__(self):
        return iter([self.landmark] if self.landmark else [])

    def __len__(self):
        return 1 if self.landmark else 0

    def __getitem__(self, idx):
        if idx == 0 and self.landmark:
            return self.landmark
        raise IndexError('Index out of range for pose_landmarks')


class MediaPipePoseLandmarkResult:
    '''Polymorphic result structure mimicking MediaPipe Solutions & Tasks API outputs.'''
    def __init__(self, landmarks: Optional[List[PoseKeypoint]]):
        self.pose_landmarks = YOLOLandmarksWrapper(landmarks) if landmarks else None
        self.landmarks = landmarks


class YOLOPoseDetector:
    '''Unified pose estimator supporting PyTorch, ONNX, and TensorRT engines.'''

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = 'auto',
        conf_threshold: float = YOLO_POSE_CONFIDENCE,
        input_size: Tuple[int, int] = YOLO_POSE_INPUT_SIZE,
        device: str = 'cuda',
    ):
        self.model_path = Path(model_path or YOLO_POSE_MODEL_PATH)
        if not self.model_path.is_file():
            raise FileNotFoundError(f'Pose weights not found: {self.model_path}')
        self.conf_threshold = float(conf_threshold)
        self.input_size = input_size
        self.device = device
        self.backend = backend
        self.last_target_center = None
        self.locked_profile = None  # Biometric and skin profile of locked assistant
        self.target_lost_frames = 0
        self._init_backend()

    def _init_backend(self):
        ext = self.model_path.suffix.lower()
        if self.backend == 'auto':
            if ext == '.engine':
                self.backend = 'tensorrt'
            elif ext == '.onnx':
                self.backend = 'onnx'
            else:
                self.backend = 'ultralytics'

        if self.backend == 'tensorrt':
            self._init_tensorrt()
        elif self.backend == 'onnx':
            self._init_onnx()
        else:
            self._init_ultralytics()

    def _init_ultralytics(self):
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))
        print(f'  [YOLOPoseDetector] Loaded Ultralytics PyTorch model: {self.model_path.name}')

    def _init_onnx(self):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f'  [YOLOPoseDetector] Loaded ONNX model with providers {self.session.get_providers()}: {self.model_path.name}')

    def _init_tensorrt(self):
        try:
            import tensorrt as trt
            import torch

            logger = trt.Logger(trt.Logger.WARNING)
            with open(str(self.model_path), 'rb') as f:
                runtime = trt.Runtime(logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.trt_inputs = []
            self.trt_outputs = []
            self.trt_bindings = []
            self.trt_tensor_buffers = []

            for binding in self.engine:
                shape = tuple(max(1, int(s)) for s in self.engine.get_binding_shape(binding))
                t = torch.empty(shape, dtype=torch.float32, device='cuda')
                self.trt_tensor_buffers.append(t)
                self.trt_bindings.append(int(t.data_ptr()))
                if self.engine.binding_is_input(binding):
                    self.trt_inputs.append(t)
                else:
                    self.trt_outputs.append(t)
            print(f'  [YOLOPoseDetector] Loaded TensorRT Engine via PyTorch CUDA bindings: {self.model_path.name}')
        except Exception as e:
            print(f'  [ERROR] Gagal inisialisasi TensorRT: {e}. Fallback ke ONNX/Ultralytics.')
            onnx_path = self.model_path.parent / 'yolov8n-pose.onnx'
            if onnx_path.exists():
                self.model_path = onnx_path
                self.backend = 'onnx'
                self._init_onnx()
            else:
                self.backend = 'ultralytics'
                self._init_ultralytics()

    def detect(self, frame: np.ndarray) -> Optional[List[PoseKeypoint]]:
        '''
        Deteksi pose pada 1 frame BGR.
        Returns:
            List of 13 PoseKeypoint objek, atau None jika tidak terdeteksi.
        '''
        if frame is None or frame.size == 0:
            return None

        if self.backend == 'ultralytics':
            return self._detect_ultralytics(frame, raw_frame=frame)
        elif self.backend == 'onnx':
            return self._detect_onnx(frame)
        elif self.backend == 'tensorrt':
            return self._detect_tensorrt(frame)
        return None

    def _detect_ultralytics(self, frame: np.ndarray, raw_frame: Optional[np.ndarray] = None) -> Optional[List[PoseKeypoint]]:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, device=self.device,
            imgsz=(self.input_size[1], self.input_size[0]), rect=True,
        )
        if not results or len(results) == 0:
            self.target_lost_frames += 1
            return None
        return self._keypoints_from_ultralytics_result(results[0], raw_frame=raw_frame if raw_frame is not None else frame)

    def reset_tracking(self):
        """Start each independent source video without a previous target."""
        self.last_target_center = None
        self.locked_profile = None
        self.target_lost_frames = 0

    def detect_batch(self, frames):
        """Batch pose inference; apply live target selection in temporal order."""
        if not frames:
            return []
        if self.backend != 'ultralytics':
            return [self.detect(frame) for frame in frames]
        results = self.model(
            frames, verbose=False, conf=self.conf_threshold, device=self.device,
            imgsz=(self.input_size[1], self.input_size[0]), rect=True,
        )
        if len(results) != len(frames):
            raise ValueError('Pose inference did not return one result per frame')
        return [self._keypoints_from_ultralytics_result(r, raw_frame=frames[i]) for i, r in enumerate(results)]

    def _keypoints_from_ultralytics_result(self, r, raw_frame: Optional[np.ndarray] = None):
        """Shared selection, normalization, confidence and anatomical checks."""
        if r.keypoints is None or len(r.keypoints.data) == 0:
            self.target_lost_frames += 1
            return None

        boxes = r.boxes
        num_candidates = len(boxes) if boxes is not None else 0
        best_idx = 0

        if num_candidates > 1:
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                cxs = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
                cys = (xyxy[:, 1] + xyxy[:, 3]) * 0.5

                # If assistant target is locked, score candidates by skin tone and spatial continuity
                if self.locked_profile is not None:
                    best_score = -999.0
                    matched_idx = 0
                    for idx in range(num_candidates):
                        cand_kpts = r.keypoints.xyn[idx].cpu().numpy()
                        score = 0.0

                        # 1. Skin tone similarity (YCrCb color space: Cr and Cb channels)
                        if raw_frame is not None and 'mean_cr' in self.locked_profile:
                            cand_skin = self._extract_skin_profile(raw_frame, cand_kpts)
                            if cand_skin is not None:
                                d_cr = abs(cand_skin['cr'] - self.locked_profile['mean_cr'])
                                d_cb = abs(cand_skin['cb'] - self.locked_profile['mean_cb'])
                                d_y = abs(cand_skin['y'] - self.locked_profile['mean_y'])
                                # Lower color distance = higher score
                                skin_score = max(0.0, 100.0 - (d_cr * 3.0 + d_cb * 3.0 + d_y * 0.5))
                                score += skin_score

                        # 2. Spatial proximity to last known assistant location
                        if self.last_target_center is not None and self.target_lost_frames < 20:
                            dist = math.hypot(cxs[idx] - self.last_target_center[0], cys[idx] - self.last_target_center[1])
                            spatial_score = max(0.0, 60.0 - (dist / 10.0))
                            score += spatial_score

                        # 3. Area consistency
                        score += min(20.0, (areas[idx] / 15000.0))

                        if score > best_score:
                            best_score = score
                            matched_idx = idx
                    best_idx = matched_idx
                elif self.last_target_center is not None and self.target_lost_frames < 15:
                    dist_sq = (cxs - self.last_target_center[0])**2 + (cys - self.last_target_center[1])**2
                    min_dist_idx = int(np.argmin(dist_sq))
                    if np.sqrt(dist_sq[min_dist_idx]) < 220.0:
                        best_idx = min_dist_idx
                    else:
                        best_idx = int(np.argmax(areas))
                else:
                    best_idx = int(np.argmax(areas))

                self.last_target_center = (float(cxs[best_idx]), float(cys[best_idx]))
                self.target_lost_frames = 0
            except Exception:
                best_idx = 0
        elif num_candidates == 1:
            best_idx = 0
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                self.last_target_center = (float((xyxy[0, 0] + xyxy[0, 2]) * 0.5), float((xyxy[0, 1] + xyxy[0, 3]) * 0.5))
                self.target_lost_frames = 0
            except Exception:
                pass

        xyn = r.keypoints.xyn[best_idx].cpu().numpy()  # (17, 2)
        conf = r.keypoints.conf[best_idx].cpu().numpy()  # (17,)

        keypoints = []
        for kpt_idx in COCO_SELECTED_KEYPOINTS:
            x = float(xyn[kpt_idx, 0])
            y = float(xyn[kpt_idx, 1])
            v = float(conf[kpt_idx])
            keypoints.append(PoseKeypoint(x, y, 0.0, v))

        # Validate anatomical structure: reject phantom detections (e.g. isolated pants fabric / chair legs)
        # Must have at least 1 shoulder AND 1 hip visible, OR (nose + shoulder), OR (nose + hip)
        has_nose = keypoints[0].visibility >= 0.25
        has_shoulder = (keypoints[1].visibility >= 0.25) or (keypoints[2].visibility >= 0.25)
        has_hip = (keypoints[7].visibility >= 0.25) or (keypoints[8].visibility >= 0.25)
        if not ((has_shoulder and has_hip) or (has_nose and has_shoulder) or (has_nose and has_hip)):
            return None

        return keypoints

    def _extract_skin_profile(self, frame: np.ndarray, xyn: np.ndarray) -> Optional[dict]:
        """Extract mean skin color in YCrCb space from facial keypoints."""
        if frame is None or frame.size == 0 or xyn is None:
            return None
        h, w = frame.shape[:2]
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # Standard human skin range in YCrCb
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        # Sample around nose (idx 0), left ear (idx 3), right ear (idx 4)
        sampled_pixels = []
        radius = max(6, int(min(w, h) * 0.035))
        for k_idx in (0, 3, 4):
            if k_idx < len(xyn):
                kx, ky = int(xyn[k_idx, 0] * w), int(xyn[k_idx, 1] * h)
                x1, x2 = max(0, kx - radius), min(w, kx + radius)
                y1, y2 = max(0, ky - radius), min(h, ky + radius)
                if x2 > x1 and y2 > y1:
                    patch_mask = skin_mask[y1:y2, x1:x2]
                    patch_ycrcb = ycrcb[y1:y2, x1:x2]
                    pts = patch_ycrcb[patch_mask > 0]
                    if len(pts) > 0:
                        sampled_pixels.append(pts)
        if sampled_pixels:
            all_pts = np.vstack(sampled_pixels)
            if len(all_pts) >= 15:
                return {
                    'mean_y': float(np.mean(all_pts[:, 0])),
                    'mean_cr': float(np.mean(all_pts[:, 1])),
                    'mean_cb': float(np.mean(all_pts[:, 2])),
                    'count': len(all_pts),
                }
        return None

    def lock_assistant_target(self, frame: np.ndarray) -> dict:
        """Register and lock the assistant currently in front of the camera."""
        if frame is None:
            return {'success': False, 'message': 'Frame kamera tidak tersedia'}
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, device=self.device,
            imgsz=(self.input_size[1], self.input_size[0]), rect=True,
        )
        if not results or len(results[0].boxes) == 0:
            return {'success': False, 'message': 'Tidak ada orang terdeteksi di depan kamera'}

        # Pick the most central, largest person
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best_idx = int(np.argmax(areas))
        kpts = results[0].keypoints.xyn[best_idx].cpu().numpy()

        skin = self._extract_skin_profile(frame, kpts)
        cx = float((xyxy[best_idx, 0] + xyxy[best_idx, 2]) * 0.5)
        cy = float((xyxy[best_idx, 1] + xyxy[best_idx, 3]) * 0.5)
        self.last_target_center = (cx, cy)
        self.target_lost_frames = 0

        profile = {
            'center': (cx, cy),
            'area': float(areas[best_idx]),
            'locked_at': time.time(),
        }
        if skin is not None:
            profile.update(skin)
            desc = f"Skin Tone YCrCb [{skin['mean_y']:.0f}, {skin['mean_cr']:.0f}, {skin['mean_cb']:.0f}]"
        else:
            desc = "Spatial Position Locked"

        self.locked_profile = profile
        return {'success': True, 'message': f'Berhasil mengunci asisten ({desc})', 'profile': profile}

    def unlock_assistant_target(self):
        """Clear locked target."""
        self.locked_profile = None
        return {'success': True, 'message': 'Kuncian target dilepas (mode deteksi bebas)'}

    def _detect_onnx(self, frame: np.ndarray) -> Optional[List[PoseKeypoint]]:
        h0, w0 = frame.shape[:2]
        inp_w, inp_h = self.input_size
        resized = cv2.resize(frame, (inp_w, inp_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: blob})
        out = outputs[0]  # shape: (1, 56, 8400)
        out = np.transpose(out[0], (1, 0))  # (8400, 56)

        # out contains [x_center, y_center, w, h, box_score, kpt0_x, kpt0_y, kpt0_conf, ...] per anchor
        scores = out[:, 4]
        best_anchor_idx = np.argmax(scores)
        if scores[best_anchor_idx] < self.conf_threshold:
            return None

        best_row = out[best_anchor_idx]
        kpts_raw = best_row[5:].reshape(17, 3)

        keypoints = []
        for kpt_idx in COCO_SELECTED_KEYPOINTS:
            x = float(kpts_raw[kpt_idx, 0] / inp_w)
            y = float(kpts_raw[kpt_idx, 1] / inp_h)
            v = float(kpts_raw[kpt_idx, 2])
            keypoints.append(PoseKeypoint(x, y, 0.0, v))
        return keypoints

    def _detect_tensorrt(self, frame: np.ndarray) -> Optional[List[PoseKeypoint]]:
        if not hasattr(self, 'context') or not self.trt_inputs:
            return self._detect_onnx(frame)
        try:
            import torch
            inp_w, inp_h = self.input_size
            resized = cv2.resize(frame, (inp_w, inp_h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor_inp = torch.from_numpy(rgb).to(device='cuda', dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            self.trt_inputs[0].copy_(tensor_inp)

            stream_handle = torch.cuda.current_stream().cuda_stream
            self.context.execute_async_v2(bindings=self.trt_bindings, stream_handle=stream_handle)
            torch.cuda.current_stream().synchronize()

            out_tensor = self.trt_outputs[0][0].permute(1, 0)  # shape: (8400, 56)

            scores = out_tensor[:, 4]
            best_anchor_idx = int(torch.argmax(scores).item())
            if float(scores[best_anchor_idx].item()) < self.conf_threshold:
                return None

            best_row = out_tensor[best_anchor_idx].cpu().numpy()
            kpts_raw = best_row[5:].reshape(17, 3)

            keypoints = []
            for kpt_idx in COCO_SELECTED_KEYPOINTS:
                x = float(kpts_raw[kpt_idx, 0] / inp_w)
                y = float(kpts_raw[kpt_idx, 1] / inp_h)
                v = float(kpts_raw[kpt_idx, 2])
                keypoints.append(PoseKeypoint(x, y, 0.0, v))
            return keypoints
        except Exception as e:
            print(f'  [ERROR] TensorRT inference error: {e}. Fallback ke ONNX.')
            return self._detect_onnx(frame)

    def process(self, frame_rgb: np.ndarray) -> MediaPipePoseLandmarkResult:
        '''MediaPipe Solutions API compatibility wrapper.'''
        if frame_rgb is None or frame_rgb.size == 0:
            return MediaPipePoseLandmarkResult(None)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3 else frame_rgb
        landmarks = self.detect(frame_bgr)
        return MediaPipePoseLandmarkResult(landmarks)

    def detect_for_video(self, mp_image: Any, timestamp_ms: int = 0) -> MediaPipePoseLandmarkResult:
        '''MediaPipe Tasks API compatibility wrapper.'''
        if mp_image is None:
            return MediaPipePoseLandmarkResult(None)
        frame_numpy = mp_image.numpy_view() if hasattr(mp_image, 'numpy_view') else mp_image
        frame_bgr = cv2.cvtColor(frame_numpy, cv2.COLOR_RGB2BGR) if frame_numpy.ndim == 3 and frame_numpy.shape[2] == 3 else frame_numpy
        landmarks = self.detect(frame_bgr)
        return MediaPipePoseLandmarkResult(landmarks)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        pass
