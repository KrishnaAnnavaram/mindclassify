"""The classifier interface. Every model returns log-scores over the same label order.

``log_scores`` are unnormalised log-probabilities (or logits). ``predict_proba`` applies a softmax
with the calibration temperature, so all models give comparable class probabilities.
"""

from __future__ import annotations

import numpy as np

from .. import LABELS


def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Classifier:
    name = "base"

    def __init__(self, labels=LABELS):
        self.labels = list(labels)
        self.temperature = 1.0

    def fit(self, texts, labels) -> "Classifier":
        raise NotImplementedError

    def log_scores(self, texts) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, texts) -> np.ndarray:
        return softmax(self.log_scores(texts), self.temperature)

    def predict(self, texts) -> list[str]:
        return [self.labels[i] for i in self.predict_proba(texts).argmax(axis=1)]


class MajorityClassifier(Classifier):
    """Always the most frequent training label. The floor for every metric."""

    name = "majority"

    def fit(self, texts, labels):
        counts = np.array([list(labels).count(l) for l in self.labels], dtype=float)
        self.prior = np.log((counts + 1) / (counts.sum() + len(counts)))
        return self

    def log_scores(self, texts):
        return np.tile(self.prior, (len(texts), 1))
