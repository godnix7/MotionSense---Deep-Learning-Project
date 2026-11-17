"""Evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
        "n_samples": int(len(y_true)),
    }
