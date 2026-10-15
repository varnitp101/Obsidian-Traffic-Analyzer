"""
src/ml/evaluate.py
------------------
Comprehensive multi-class model evaluation module for UK Traffic Accident Severity.
Focuses on:
1. Macro F1 (equally weighting all 3 classes).
2. Minority Recall (Class 0 Fatal and Class 1 Serious recall).
3. Precision-Recall AUC and Confusion Matrix visualization.
"""

import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)

logger = logging.getLogger("ml.evaluate")

CLASS_NAMES = ["Fatal (1)", "Serious (2)", "Slight (3)"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    prefix: str = "test",
) -> Dict[str, float]:
    """
    Computes holistic classification metrics tailored for imbalanced multi-class problems.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    
    # Per-class recall and precision
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        f"{prefix}_macro_f1": macro_f1,
        f"{prefix}_weighted_f1": weighted_f1,
        f"{prefix}_fatal_recall": float(recalls[0]),
        f"{prefix}_serious_recall": float(recalls[1]),
        f"{prefix}_slight_recall": float(recalls[2]),
        f"{prefix}_fatal_precision": float(precisions[0]),
        f"{prefix}_serious_precision": float(precisions[1]),
        f"{prefix}_slight_precision": float(precisions[2]),
        f"{prefix}_fatal_f1": float(f1s[0]),
        f"{prefix}_serious_f1": float(f1s[1]),
        f"{prefix}_slight_f1": float(f1s[2]),
    }

    if y_prob is not None:
        try:
            metrics[f"{prefix}_log_loss"] = float(log_loss(y_true, y_prob))
            metrics[f"{prefix}_roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception as e:
            logger.warning(f"Could not compute probability-based metrics: {e}")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    normalize: bool = True,
    title: str = "Normalized Confusion Matrix",
) -> str:
    """
    Generates and saves a clean confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        display_data = cm_norm
        fmt = ".2%"
    else:
        display_data = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(display_data, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title=title,
        ylabel="True Severity",
        xlabel="Predicted Severity",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = display_data.max() / 2.0
    for i in range(display_data.shape[0]):
        for j in range(display_data.shape[1]):
            val_str = f"{display_data[i, j]:{fmt}}"
            ax.text(
                j, i, val_str,
                ha="center", va="center",
                color="white" if display_data[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
