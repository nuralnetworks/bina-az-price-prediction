"""Metrics in NumPy + a small comparison printer."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    e = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def r2(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - np.sum((y - p) ** 2) / ss_tot)


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return np.array([[np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1))],
                     [np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1))]])


def precision_recall_f1(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return float(prec), float(rec), float(f1)


def roc_auc(y_true, scores) -> float:
    y = np.asarray(y_true).ravel().astype(float)
    s = np.asarray(scores).ravel().astype(float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def compare(name_to_metrics: dict) -> None:
    names = list(name_to_metrics)
    keys = []
    for n in names:
        for k in name_to_metrics[n]:
            if k not in keys:
                keys.append(k)
    widths = [max(len(n) for n in names + ["model"])] + [max(len(k), 12) for k in keys]
    print(" | ".join(["model".ljust(widths[0])] + [k.ljust(w) for k, w in zip(keys, widths[1:])]))
    print("-+-".join(["-" * w for w in widths]))
    for n in names:
        row = [n.ljust(widths[0])]
        for k, w in zip(keys, widths[1:]):
            v = name_to_metrics[n].get(k, "-")
            row.append((f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(w))
        print(" | ".join(row))
