"""Synthetic posts for tests and the offline demo. They are template sentences, not real posts.

The generator copies the shape of the public corpus: 7 labels with class imbalance, two sources
with different styles, shared words between near classes (Depression and Suicidal, Anxiety and
Stress), 5% label noise, exact copies and near copies. It never uses text from a real person.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import LABELS

SHARE = {"Normal": 0.30, "Depression": 0.27, "Suicidal": 0.18, "Anxiety": 0.09, "Bipolar": 0.06,
         "Stress": 0.06, "Personality disorder": 0.04}

CUES = {
    "Normal": ["weekend", "football", "recipe", "garden", "movie", "holiday", "coffee", "concert", "cousin"],
    "Depression": ["empty", "numb", "hopeless", "tired", "worthless", "grey", "alone", "nothing matters"],
    "Suicidal": ["end it", "no way out", "goodbye note", "not be here", "better without me", "final plan"],
    "Anxiety": ["panic", "racing heart", "worry", "nervous", "dread", "shaking", "what if"],
    "Bipolar": ["manic", "no sleep for days", "highs and lows", "racing ideas", "mood swing", "lithium"],
    "Stress": ["deadline", "workload", "exams", "bills", "pressure", "overtime", "too many tasks"],
    "Personality disorder": ["abandonment", "unstable identity", "split", "intense relationships", "emptiness"],
}
NEAR = {"Depression": "Suicidal", "Suicidal": "Depression", "Anxiety": "Stress", "Stress": "Anxiety",
        "Bipolar": "Depression", "Personality disorder": "Depression", "Normal": "Stress"}
FILLER = ["today", "I", "feel", "my", "and", "the", "again", "really", "this", "week", "about", "so", "it", "is",
          "with", "at", "night", "work", "friends", "family"]


@dataclass(frozen=True)
class SynthSpec:
    n: int = 4000
    label_noise: float = 0.05
    exact_dup_rate: float = 0.04
    near_dup_rate: float = 0.03
    seed: int = 42


def _post(rng, label: str, style: str) -> str:
    own = list(rng.choice(CUES[label], size=rng.integers(1, 3)))
    other = [str(rng.choice(CUES[NEAR[label]]))] if rng.uniform() < 0.35 else []
    words = list(rng.choice(FILLER, size=rng.integers(6, 18))) + own + other
    rng.shuffle(words)
    text = " ".join(words)
    if style == "forum_b":
        text = text.lower().replace(" you ", " u ") + (" idk" if rng.uniform() < 0.3 else "")
    else:
        text = text[0].upper() + text[1:] + "."
    return text


def generate(spec: SynthSpec = SynthSpec()) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    labels = rng.choice(list(SHARE), size=spec.n, p=np.array(list(SHARE.values())))
    rows = []
    for i, lab in enumerate(labels):
        src = "forum_a" if rng.uniform() < 0.6 else "forum_b"
        text = _post(rng, lab, src)
        shown = str(rng.choice(LABELS)) if rng.uniform() < spec.label_noise else lab
        rows.append((f"s{i}", text, shown, src))
    df = pd.DataFrame(rows, columns=["id", "statement", "status", "source"])
    n_exact = int(spec.exact_dup_rate * spec.n)
    n_near = int(spec.near_dup_rate * spec.n)
    exact = df.sample(n_exact, random_state=spec.seed).copy()
    near = df.sample(n_near, random_state=spec.seed + 1).copy()
    near["statement"] = near["statement"] + " really"
    extra = pd.concat([exact, near], ignore_index=True)
    extra["id"] = [f"d{i}" for i in range(len(extra))]
    out = pd.concat([df, extra], ignore_index=True).sample(frac=1.0, random_state=spec.seed).reset_index(drop=True)
    out["status"] = out["status"].replace({"Bipolar": "Bi-Polar"}).where(rng.uniform(size=len(out)) < 0.5,
                                                                         out["status"])
    return out
