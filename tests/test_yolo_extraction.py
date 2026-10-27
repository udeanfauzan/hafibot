"""Regression coverage for offline/live pose parity and cache provenance."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

from src.config import training_dataset_sources
from src.extract_features_yolo import (
    extract_landmarks_from_video, landmarks_to_features, process_dataset, write_metadata,
)
from src.pose_provenance import pose_extraction_signature, validate_pose_metadata
from src.yolo_detector import YOLOPoseDetector


class Boxes:
    def __init__(self, boxes):
        self.xyxy = torch.tensor(boxes, dtype=torch.float32)

    def __len__(self):
        return len(self.xyxy)


def result(boxes, confidence=0.9):
    count = len(boxes)
    joints = torch.zeros((count, 17, 2), dtype=torch.float32)
    for index in range(count):
        joints[index, :, 0] = 0.2 + index * 0.5
        joints[index, :, 1] = 0.3
    return SimpleNamespace(
        boxes=Boxes(boxes),
        keypoints=SimpleNamespace(
            data=torch.zeros((count, 17, 3)), xyn=joints,
            conf=torch.full((count, 17), confidence),
        ),
    )


def detector_with_results(results):
    detector = YOLOPoseDetector.__new__(YOLOPoseDetector)
    detector.backend = "ultralytics"
    detector.device = "cpu"
    detector.conf_threshold = 0.35
    detector.input_size = (640, 640)
    detector.reset_tracking()
    queue = list(results)

    def predict(frames, **kwargs):
        count = len(frames) if isinstance(frames, list) else 1
        selected = queue[:count]
        del queue[:count]
        return selected

    detector.model = mock.Mock(side_effect=predict)
    return detector


class YOLOExtractionTests(unittest.TestCase):
    def test_batched_extraction_matches_live_selection_across_larger_bystander(self):
        results = [
            result([[40, 20, 140, 220]]),
            result([[45, 20, 145, 220], [280, 5, 630, 470]]),
            result([]),
            result([[50, 20, 150, 220], [280, 5, 630, 470]]),
        ]
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in results]
        live = detector_with_results(results)
        expected = np.asarray([landmarks_to_features(live.detect(frame)) for frame in frames])
        offline = detector_with_results(results)
        actual = np.asarray([landmarks_to_features(pose) for pose in offline.detect_batch(frames)])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(float(actual[1, 0]), float(actual[0, 0]))
        np.testing.assert_array_equal(actual[2], 0)
        live_kwargs = live.model.call_args.kwargs
        self.assertEqual(live_kwargs, offline.model.call_args.kwargs)

    def test_low_confidence_phantom_is_rejected_offline_and_live(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        live = detector_with_results([result([[0, 0, 100, 100]], confidence=0.1)])
        offline = detector_with_results([result([[0, 0, 100, 100]], confidence=0.1)])
        self.assertIsNone(live.detect(frame))
        self.assertEqual(offline.detect_batch([frame]), [None])

    def test_new_video_does_not_inherit_previous_target(self):
        detector = detector_with_results([
            result([[40, 20, 140, 220]]),
            result([[45, 20, 145, 220], [280, 5, 630, 470]]),
        ])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame)
        detector.reset_tracking()
        pose = detector.detect(frame)
        self.assertGreater(pose[0].x, 0.5)

    def test_assistant_target_lock_prefers_matching_profile_over_nearer_person(self):
        # Candidate 0: nearer to previous center (100, 100), but has different profile
        # Candidate 1: slightly farther, but matches locked target profile
        detector = detector_with_results([
            result([[40, 20, 140, 220], [280, 5, 630, 470]]),
        ])
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.last_target_center = (455.0, 237.5)
        detector.locked_profile = {'mean_cr': 145.0, 'mean_cb': 110.0, 'mean_y': 160.0}
        # Mock _extract_skin_profile: cand 0 gives wrong skin, cand 1 gives exact matching skin
        detector._extract_skin_profile = mock.Mock(side_effect=[
            {'cr': 120.0, 'cb': 90.0, 'y': 100.0},
            {'cr': 145.0, 'cb': 110.0, 'y': 160.0},
        ])
        pose = detector.detect(frame)
        # Candidate 1 was selected (index 1 -> x = 0.2 + 0.5 = 0.7)
        self.assertGreater(pose[0].x, 0.5)

    def test_extraction_preserves_missing_frames_and_final_partial_batch(self):
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 3
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda key: {
            cv2.CAP_PROP_FPS: 15.0, cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }[key]
        cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        detector = detector_with_results([
            result([[0, 0, 100, 200]]), result([]), result([[0, 0, 100, 200]]),
        ])
        with mock.patch("src.extract_features_yolo.cv2.VideoCapture", return_value=cap):
            values, info = extract_landmarks_from_video(Path("clip.mp4"), detector, batch_size=2)
        self.assertEqual(values.shape, (3, 52))
        self.assertEqual(info["detected_frames"], 2)
        np.testing.assert_array_equal(values[1], 0)
        cap.release.assert_called_once()

    def test_wrong_fps_fails_before_inference(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30.0
        detector = detector_with_results([])
        with mock.patch("src.extract_features_yolo.cv2.VideoCapture", return_value=cap):
            with self.assertRaisesRegex(ValueError, "resample real motion"):
                extract_landmarks_from_video(Path("clip.mp4"), detector)
        detector.model.assert_not_called()
        cap.release.assert_called_once()

    def test_provenance_rejects_changed_weights_and_incomplete_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "yolo11m-pose.pt"
            weights.write_bytes(b"first weights")
            signature = pose_extraction_signature(weights)
            metadata = {"schema_version": 2, **signature, "completed": True}
            validate_pose_metadata(metadata, signature)
            weights.write_bytes(b"different weights")
            with self.assertRaisesRegex(ValueError, "model_sha256"):
                validate_pose_metadata(metadata, pose_extraction_signature(weights))
            metadata["completed"] = False
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_pose_metadata(metadata, signature)

    def test_legacy_cache_cannot_be_relabelled_as_yolo11(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for category in ("ADL", "Fall"):
                (root / "raw" / category).mkdir(parents=True)
                (root / "raw" / category / "video.mp4").write_bytes(b"video")
            output = root / "output"
            output.mkdir()
            original = {"schema_version": 1, "extractor": "yolov8n-pose"}
            metadata_path = output / "extraction_metadata.json"
            metadata_path.write_text(json.dumps(original))
            weights = root / "yolo11m-pose.pt"
            weights.write_bytes(b"weights")
            detector = SimpleNamespace(model_path=weights, conf_threshold=0.35, input_size=(640, 640))
            with self.assertRaisesRegex(ValueError, "schema version 2"):
                process_dataset(root / "raw", output, detector)
            self.assertEqual(json.loads(metadata_path.read_text()), original)

    def test_isolated_cache_preserves_source_identity_and_annotations(self):
        default = training_dataset_sources()
        isolated = training_dataset_sources(Path("data/processed_yolo11m"))
        for before, after in zip(default, isolated):
            self.assertEqual(before["name"], after["name"])
            self.assertEqual(before["fall_annotations_path"], after["fall_annotations_path"])
            self.assertNotEqual(before["processed_adl_dir"], after["processed_adl_dir"])

    def test_metadata_write_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extraction_metadata.json"
            path.write_text('{"completed": false}')
            original_replace = Path.replace
            attempts = []

            def locked_once(temporary, target):
                attempts.append(target)
                if len(attempts) == 1:
                    self.assertFalse(json.loads(path.read_text())["completed"])
                    raise PermissionError("Windows reader has the manifest open")
                return original_replace(temporary, target)

            with mock.patch.object(Path, "replace", autospec=True, side_effect=locked_once):
                with mock.patch("src.extract_features_yolo.time.sleep") as sleep:
                    write_metadata(path, {"completed": True})
            self.assertEqual(len(attempts), 2)
            sleep.assert_called_once_with(0.1)
            self.assertTrue(json.loads(path.read_text())["completed"])


if __name__ == "__main__":
    unittest.main()
