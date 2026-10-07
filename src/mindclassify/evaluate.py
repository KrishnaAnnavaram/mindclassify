"""Metrics (macro-F1 first), bootstrap intervals, calibration, slices and the safety rule.

- Primary metric: macro-F1 over all labels. Accuracy is reported but never alone.
- Calibration: temperature scaling fit on the VALIDATION split (minimum negative log-likelihood),
  and the expected calibration error (ECE, 15 bins) before and after.
- Safety rule: if P(Suicidal) >= t, the post is routed to a human reviewer, whatever the top label.
  t is the highest threshold that reaches the target recall on the validation split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from . import SAFETY_LABEL
from .models.base import softmax
from .safety import route
from .splits import length_bucket


def _idx(labels, y) -> np.ndarray:
    pos = {l: i for i, l in enumerate(labels)}
    return np.array([pos[v] for v in y])


def fit_temperature(log_scores: np.ndarray, y_idx: np.ndarray) -> float:
    def nll(t):
        p = softmax(log_scores, t)
        return -np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12))

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def ece(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_idx
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def safety_threshold(p_safety: np.ndarray, is_safety: np.ndarray, target_recall: float) -> float:
    """The highest threshold whose recall on these posts is >= target_recall."""
    pos = np.sort(p_safety[is_safety])[::-1]
    if pos.size == 0:
        raise ValueError(f"no {SAFETY_LABEL} posts to set the threshold")
    k = int(np.ceil(target_recall * pos.size)) - 1
    return float(pos[max(0, k)])


def _routing(routed: np.ndarray, is_safety: np.ndarray) -> dict:
    tp = int((routed & is_safety).sum())
    return {"recall": tp / max(1, int(is_safety.sum())), "precision": tp / max(1, int(routed.sum())),
            "routed_share": float(routed.mean())}


def safety_metrics(p_safety: np.ndarray, is_safety: np.ndarray, threshold: float, texts=None) -> dict:
    """Recall, precision and routed share of the threshold rule, and of threshold OR crisis phrase."""
    out = {"threshold": threshold, **_routing(p_safety >= threshold, is_safety)}
    if texts is not None:
        out["with_phrases"] = _routing(route(p_safety, texts, threshold), is_safety)
    return out


def macro_f1_ci(y: np.ndarray, pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], pred[i], labels=labels, average="macro", zero_division=0))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def metrics(labels, y_true, probs: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    y = _idx(labels, y_true)
    pred = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(labels)), zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, labels=range(len(labels)), average="macro", zero_division=0)),
        "macro_f1_ci95": macro_f1_ci(y, pred, n_boot, seed),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "ece": ece(probs, y),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion": confusion_matrix(y, pred, labels=range(len(labels))).tolist(),
    }


def slices(labels, df: pd.DataFrame, probs: np.ndarray) -> list[dict]:
    y = _idx(labels, df["label"])
    pred = probs.argmax(axis=1)
    out = []
    for name, keys in (("source", df["source"].astype(str)), ("length", length_bucket(df["text"]).astype(str))):
        for k in sorted(keys.unique()):
            m = (keys == k).to_numpy()
            out.append({"slice": name, "value": k, "n": int(m.sum()),
                        "macro_f1": float(f1_score(y[m], pred[m], average="macro", zero_division=0))})
    return out


def calibrate_and_route(model, val_df: pd.DataFrame, target_recall: float) -> dict:
    """Fit the temperature and the safety threshold on the validation split. Sets ``model.temperature``."""
    z = model.log_scores(val_df["text"].tolist())
    y = _idx(model.labels, val_df["label"])
    before = ece(softmax(z, 1.0), y)
    model.temperature = fit_temperature(z, y)
    probs = softmax(z, model.temperature)
    out = {"temperature": model.temperature, "val_ece_before": before, "val_ece_after": ece(probs, y)}
    if SAFETY_LABEL in model.labels:
        k = model.labels.index(SAFETY_LABEL)
        out["safety_threshold"] = safety_threshold(probs[:, k], y == k, target_recall)
    return out


def evaluate(model, df: pd.DataFrame, safety_t: float | None, n_boot: int = 1000, seed: int = 0) -> dict:
    probs = model.predict_proba(df["text"].tolist())
    res = {"model": model.name, **metrics(model.labels, df["label"], probs, n_boot, seed),
           "slices": slices(model.labels, df, probs)}
    if safety_t is not None and SAFETY_LABEL in model.labels:
        k = model.labels.index(SAFETY_LABEL)
        res["safety"] = safety_metrics(probs[:, k], _idx(model.labels, df["label"]) == k, safety_t,
                                       df["text"].tolist())
    return res
