"""Stratified, group-aware train/validation/test splits with saved IDs and a leakage check."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .dedup import normalise


class LeakageError(AssertionError):
    pass


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def check(self) -> None:
        sets = {k: getattr(self, k) for k in ("train", "val", "test")}
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if set(sets[a]["group"]) & set(sets[b]["group"]):
                    raise LeakageError(f"a duplicate group is in {a} and {b}")
                if set(sets[a]["text"].map(normalise)) & set(sets[b]["text"].map(normalise)):
                    raise LeakageError(f"the same normalised text is in {a} and {b}")

    def ids(self) -> dict[str, list[str]]:
        return {k: getattr(self, k)["id"].tolist() for k in ("train", "val", "test")}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.ids()), encoding="utf-8")
        return path

    @classmethod
    def from_ids(cls, df: pd.DataFrame, path: str | Path) -> "Split":
        ids = json.loads(Path(path).read_text(encoding="utf-8"))
        by_id = df.set_index("id")
        return cls(*(by_id.loc[ids[k]].reset_index() for k in ("train", "val", "test")))


def _carve(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take about ``fraction`` of the rows (whole groups, stratified by label)."""
    n_splits = max(2, int(round(1 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, part = next(sgkf.split(df, df["label"], df["group"]))
    return df.iloc[rest].reset_index(drop=True), df.iloc[part].reset_index(drop=True)


def make_split(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42) -> Split:
    if "group" not in df:
        df = df.assign(group=df["id"])
    rest, test = _carve(df, test_size, seed)
    train, val = _carve(rest, val_size / (1 - test_size), seed + 1)
    split = Split(train, val, test)
    split.check()
    return split


def label_table(split: Split, labels) -> pd.DataFrame:
    return pd.DataFrame({k: getattr(split, k)["label"].value_counts().reindex(labels).fillna(0).astype(int)
                         for k in ("train", "val", "test")})


def length_bucket(texts: pd.Series) -> pd.Series:
    words = texts.str.split().str.len()
    return pd.cut(words, [0, 20, 60, np.inf], labels=["short (<=20 words)", "medium (21-60)", "long (>60)"])
