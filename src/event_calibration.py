"""Validation-only threshold calibration for video-level Fall events."""

import numpy as np


def event_threshold_rows(y_true, max_fall_probabilities, thresholds=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(max_fall_probabilities, dtype=np.float64)
    if y_true.shape != probabilities.shape:
        raise ValueError("Event labels/probabilities harus memiliki shape sama")
    if set(np.unique(y_true).tolist()) != {0, 1}:
        raise ValueError("Calibration set harus memiliki event ADL dan Fall")
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 1.00, 0.01), 2)

    rows = []
    for threshold in thresholds:
        prediction = (probabilities >= threshold).astype(np.int64)
        tn = int(np.sum((y_true == 0) & (prediction == 0)))
        fp = int(np.sum((y_true == 0) & (prediction == 1)))
        fn = int(np.sum((y_true == 1) & (prediction == 0)))
        tp = int(np.sum((y_true == 1) & (prediction == 1)))
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        specificity = tn / float(tn + fp) if tn + fp else 0.0
        rows.append({
            "threshold": float(threshold),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": 0.5 * (recall + specificity),
        })
    return rows


def select_safety_first_threshold(rows):
    """Maximize Fall recall, then minimize false ADL, then threshold."""
    if not rows:
        raise ValueError("Tidak ada threshold candidate")
    return max(
        rows,
        key=lambda row: (row["recall"], -row["fp"], row["threshold"]),
    )
