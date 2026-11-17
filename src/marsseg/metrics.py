"""Confusion-matrix based segmentation metrics. Ignored pixels (255) never count."""

import numpy as np

from . import CLASSES, IGNORE


def confusion(pred: np.ndarray, label: np.ndarray, n: int = len(CLASSES)) -> np.ndarray:
    """cm[t, p] = number of pixels whose true class is t and predicted class is p."""
    ok = label != IGNORE
    return np.bincount(n * label[ok].astype(int) + pred[ok].astype(int), minlength=n * n).reshape(n, n)


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    """Intersection over Union: overlap / (overlap + both kinds of mistake)."""
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1) - tp
    return np.divide(tp, denom, out=np.full(len(tp), np.nan), where=denom > 0)


def summary(cm: np.ndarray) -> dict:
    iou = iou_per_class(cm)
    return {
        "mIoU": round(float(np.nanmean(iou)), 4),
        "pixel_accuracy": round(float(np.diag(cm).sum() / cm.sum()), 4),
        "iou": {c: round(float(v), 4) for c, v in zip(CLASSES, iou)},
        "recall": {c: round(float(cm[i, i] / max(cm[i].sum(), 1)), 4) for i, c in enumerate(CLASSES)},
        "precision": {c: round(float(cm[i, i] / max(cm[:, i].sum(), 1)), 4) for i, c in enumerate(CLASSES)},
        "pixel_share": {c: round(float(cm[i].sum() / cm.sum()), 4) for i, c in enumerate(CLASSES)},
    }
