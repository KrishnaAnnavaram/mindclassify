"""Prepare the data the same way for every command: load -> clean -> de-duplicate -> split."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import LABELS
from .clean import CleanConfig, clean
from .config import Settings
from .data import LoadReport, load_csv, load_frame
from .dedup import DedupReport, group_duplicates, resolve
from .splits import Split, make_split
from .synthetic import SynthSpec, generate


@dataclass
class Prepared:
    data: pd.DataFrame
    split: Split
    load_report: LoadReport
    dedup_report: DedupReport
    removed_by_dedup: int
    synthetic: bool


def prepare(s: Settings, synthetic: bool = False, clean_cfg: CleanConfig = CleanConfig(),
            spec: SynthSpec | None = None) -> Prepared:
    if synthetic:
        df, rep = load_frame(generate(spec or SynthSpec(seed=s.seed)), "synthetic")
    else:
        df, rep = load_csv(s.data_path)
    df["text"] = df["text"].map(lambda t: clean(t, clean_cfg))
    df = df[df["text"] != ""]
    grouped, drep = group_duplicates(df, s.near_dup_threshold)
    unique = resolve(grouped)
    split = make_split(unique, s.test_size, s.val_size, s.seed)
    return Prepared(unique, split, rep, drep, len(grouped) - len(unique), synthetic)


def labels_present(p: Prepared) -> list[str]:
    """The fixed label order, limited to labels that occur in the training split."""
    seen = set(p.split.train["label"])
    return [l for l in LABELS if l in seen]
