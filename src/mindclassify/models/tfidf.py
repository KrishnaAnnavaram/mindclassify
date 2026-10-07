"""TF-IDF (word 1-2 grams + character 3-5 grams) with a class-balanced logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .base import Classifier


class TfidfClassifier(Classifier):
    name = "tfidf"

    def __init__(self, labels=None, C: float = 4.0, seed: int = 42, **kw):
        super().__init__(**({"labels": labels} if labels else {}))
        self.C, self.seed = C, seed

    def fit(self, texts, labels):
        unknown = set(labels) - set(self.labels)
        if unknown:
            raise ValueError(f"labels not in the label set: {sorted(unknown)}")
        self.pipe = Pipeline([
            ("features", FeatureUnion([
                ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)),
                ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                                          max_features=100_000)),
            ])),
            ("clf", LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000, random_state=self.seed)),
        ])
        self.pipe.fit(list(texts), list(labels))
        return self

    def log_scores(self, texts):
        logp = self.pipe.predict_log_proba(list(texts))
        out = np.full((len(texts), len(self.labels)), -50.0)  # a label absent in training gets ~0 probability
        for j, lab in enumerate(self.pipe.classes_):
            out[:, self.labels.index(lab)] = logp[:, j]
        return out
