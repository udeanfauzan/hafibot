"""Selection isolation, onset metrics, memo equivalence, and opt-in integrity."""

import json
from unittest import mock

import numpy as np
import pytest

from scripts.calibrate_runtime import calibrate, operating_point_key
from scripts.evaluate_runtime_replay import PredictionMemo, replay_records, select_replay_indices, summarize_events
from src.runtime_loader import create_server_fall_detector, verify_runtime_calibration
from src.runtime_candidate import RUNTIME_CODE_FILES, runtime_settings
from src.temporal_labels import file_sha256


class ScoreDetector:
    checkpoint = {"data_split": {"strategy": "video_stratified", "validation_videos": ["ADL", "Fall"]}}

    def __init__(self, *args, **kwargs):
        self.confidence_threshold = .5
        self.calls = 0

    def reset_buffer(self):
        pass

    def _predict(self, sequence):
        self.calls += 1
        prob = .8 if sequence[0, 0] else .2
        return 1, prob, np.array([1 - prob, prob])

    def process_frame(self, landmarks, **kwargs):
        sequence = np.full((30, 130), landmarks[0].x)
        prob = self._predict(sequence)[2][1]
        return {"is_fall": prob >= self.confidence_threshold, "model_evaluated": True,
                "posture_gated": False, "pose_detected": True, "buffer_reprimed": False}


def records(tmp_path):
    result = []
    for label, name in ((0, "ADL"), (1, "Fall")):
        values = np.ones((10, 52), dtype=np.float32)
        values[:, 0] = 0
        if label:
            values[3:7, 0] = 1
        path = tmp_path / (name + ".npy")
        np.save(path, values)
        result.append({"video": name, "label": label, "path": path, "fps": 15.,
                       "frame_size": (640, 480), "array_sha256": name, "source_sha256": name,
                       "annotation": {"fall_start_frame": 3, "fall_end_frame": 6} if label else None})
    result.append({"video": "forbidden_test", "label": 1, "path": tmp_path / "must_not_load.npy"})
    return result


def test_memo_matches_uncached_replay_and_never_caches_decisions(tmp_path):
    data = records(tmp_path)
    d = ScoreDetector()
    expected = replay_records(d, data, [0, 1], quiet=True)
    memo = PredictionMemo()
    assert expected == replay_records(d, data, [0, 1], memo=memo, quiet=True)
    calls = memo.model_calls
    d.confidence_threshold = .9
    cached = replay_records(d, data, [0, 1], memo=memo, quiet=True)
    assert cached == replay_records(d, data, [0, 1], quiet=True)
    assert memo.model_calls == calls
    assert expected[1]["onset_detected"] and not cached[1]["onset_detected"]


def test_memo_refuses_a_changed_input_under_the_same_frame_key():
    memo = PredictionMemo()
    predictor = mock.Mock(return_value=(0, .9, np.array([.9, .1])))
    memo.predict(("clip", 1), np.zeros((30, 130)), predictor)
    with pytest.raises(ValueError, match="cache is invalid"):
        memo.predict(("clip", 1), np.ones((30, 130)), predictor)
    assert predictor.call_count == 1


def test_prefall_latch_overlap_is_not_fresh_onset_recall():
    metrics = summarize_events([{"label": 1, "detected": True, "onset_detected": False,
                                "pre_fall_alert_episodes": 1}])
    assert metrics["fall_recall"] == 1 and metrics["onset_fall_recall"] == 0
    assert metrics["pre_fall_alert_episodes"] == 1


@pytest.mark.parametrize("alert_start, expected_overlap, expected_late", [(1, True, False), (7, False, True)])
def test_replay_separates_premature_latch_and_late_alerts(tmp_path, alert_start, expected_overlap, expected_late):
    class LatchDetector(ScoreDetector):
        def reset_buffer(self):
            self.index = 0

        def process_frame(self, landmarks, **kwargs):
            index = self.index
            self.index += 1
            return {"is_fall": alert_start <= index < alert_start + 5, "model_evaluated": True,
                    "posture_gated": False, "pose_detected": True, "buffer_reprimed": False}

    detail = replay_records(LatchDetector(), records(tmp_path), [1], quiet=True)[0]
    assert detail["detected"] is expected_overlap
    assert not detail["onset_detected"]
    assert detail["late_only"] is expected_late
    assert detail["alert_onset_frames"] == [alert_start]


def test_full_runtime_calibration_reads_validation_not_test_and_freezes_output(tmp_path):
    data = records(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    output = tmp_path / "selection.json"
    with mock.patch("scripts.calibrate_runtime.make_runtime_detector", ScoreDetector), \
         mock.patch("scripts.calibrate_runtime.load_replay_records", return_value=(data, tmp_path)), \
         mock.patch("scripts.evaluate_runtime_replay.split_video_indices", return_value=([], [0, 1], [2])), \
         mock.patch("scripts.calibrate_runtime.pose_extraction_signature", return_value={"test": True}):
        report = calibrate(checkpoint, tmp_path, output, profiles=("legacy_v1",))
        with pytest.raises(FileExistsError):
            calibrate(checkpoint, tmp_path, output, profiles=("legacy_v1",))
    assert report["selected"]["threshold"] == .8
    assert report["selected"]["onset_fall_recall"] == 1
    assert [r["video"] for r in report["validation_manifest"]] == ["ADL", "Fall"]
    assert json.loads(output.read_text())["selected"] == report["selected"]


def test_split_must_match_checkpoint_manifest(tmp_path):
    with mock.patch("scripts.evaluate_runtime_replay.split_video_indices", return_value=([], [1, 0], [2])):
        with pytest.raises(ValueError, match="split manifest"):
            select_replay_indices(records(tmp_path), ScoreDetector.checkpoint, "validation")


def test_selection_prioritizes_real_onsets_over_false_alarm_latch():
    base = {"onset_fall_recall": .8, "false_adl_videos": 0, "false_adl_alert_episodes": 0,
            "pre_fall_alert_episodes": 0, "median_onset_delay_sec": .3, "threshold": .8, "profile": "temporal_v2"}
    worse = {**base, "onset_fall_recall": .7, "fall_recall": 1.0}
    assert operating_point_key(base) > operating_point_key(worse)
    assert operating_point_key(base) > operating_point_key({**base, "pre_fall_alert_episodes": 1})


def test_server_default_still_constructs_original_legacy_detector():
    with mock.patch("src.runtime_loader.FallDetector") as legacy, mock.patch("src.runtime_loader.verify_runtime_calibration") as verify:
        create_server_fall_detector()
        legacy.assert_called_once_with(confidence_threshold=.9, enable_posture_filter=True)
        verify.assert_not_called()


def test_optin_refuses_modified_runtime_code(tmp_path):
    path = tmp_path / "modified.json"
    path.write_text(json.dumps({"schema_version": 1, "validation_manifest": [{}],
                               "selected": {"profile": "temporal_v2", "threshold": .8},
                               "code_sha256": {}, "fixed_settings": {}}))
    with pytest.raises(ValueError, match="recalibrate"):
        verify_runtime_calibration(path)


def test_optin_loads_exact_frozen_selection_and_refuses_changed_weights(tmp_path):
    from src.runtime_loader import PROJECT_ROOT
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"candidate")
    report = {"schema_version": 1, "validation_manifest": [{}],
              "selected": {"profile": "temporal_v2", "threshold": .8},
              "code_sha256": {name: file_sha256(PROJECT_ROOT / name) for name in RUNTIME_CODE_FILES},
              "fixed_settings": runtime_settings(), "pose_signature": {"fixed": True},
              "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    with mock.patch("src.runtime_loader.pose_extraction_signature", return_value={"fixed": True}), \
         mock.patch("src.runtime_loader.make_runtime_detector") as factory:
        create_server_fall_detector(path)
        factory.assert_called_once_with("temporal_v2", checkpoint_path=str(checkpoint),
                                        confidence_threshold=.8, enable_posture_filter=True)
        checkpoint.write_bytes(b"changed candidate")
        with pytest.raises(ValueError, match="checkpoint changed"):
            verify_runtime_calibration(path)
