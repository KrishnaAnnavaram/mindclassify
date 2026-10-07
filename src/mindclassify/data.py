"""Load posts into one schema: ``id``, ``text``, ``label``, ``source``. Labels are normalised."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import LABELS


class SchemaError(ValueError):
    pass


_LABEL_MAP = {lab.lower(): lab for lab in LABELS} | {"bi-polar": "Bipolar", "bi polar": "Bipolar",
                                                      "personality": "Personality disorder"}


def normalise_label(raw) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return _LABEL_MAP.get(str(raw).strip().lower())


@dataclass
class LoadReport:
    rows_in: int = 0
    empty_text: int = 0
    unknown_label: int = 0
    unknown_values: dict = field(default_factory=dict)
    rows_out: int = 0


def load_frame(raw: pd.DataFrame, name: str, text_col: str = "statement", label_col: str = "status",
               source_col: str | None = "source") -> tuple[pd.DataFrame, LoadReport]:
    """Normalise one raw table into ``id``, ``text``, ``label``, ``source``. Bad rows are counted and dropped."""
    missing = [c for c in (text_col, label_col) if c not in raw.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")
    rep = LoadReport(rows_in=len(raw))
    has_source = source_col is not None and source_col in raw.columns
    df = pd.DataFrame({
        "text": raw[text_col].astype("string"),
        "label": raw[label_col].map(normalise_label),
        "source": raw[source_col].astype(str) if has_source else name,
    })
    empty = df["text"].isna() | (df["text"].str.strip() == "")
    rep.empty_text = int(empty.sum())
    unknown = df["label"].isna() & ~empty
    rep.unknown_label = int(unknown.sum())
    rep.unknown_values = raw.loc[unknown, label_col].astype(str).value_counts().head(10).to_dict()
    df = df[~empty & ~unknown].copy()
    df["text"] = df["text"].astype(str)
    df.insert(0, "id", [f"{name}-{i}" for i in df.index])
    rep.rows_out = len(df)
    return df.reset_index(drop=True), rep


def load_csv(path: str | Path, **kw) -> tuple[pd.DataFrame, LoadReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    return load_frame(pd.read_csv(path), path.stem.replace(" ", "_"), **kw)
