"""Exact and near-duplicate groups (MinHash with LSH bands), joined by union-find.

Every post gets a ``group`` ID. Posts in one group are copies or near copies, so the split keeps a
whole group on one side. A group with two different labels is reported as a label conflict.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

_NORM = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    return " ".join(_NORM.sub(" ", text.lower()).split())


def shingles(text: str, k: int = 3) -> set[str]:
    words = normalise(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class DedupReport:
    posts: int
    exact_duplicates: int
    near_duplicate_links: int
    groups: int
    label_conflict_groups: int


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser on uint64 arrays (wrap-around arithmetic)."""
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def minhash_signatures(texts, num_perm: int = 64, seed: int = 1) -> np.ndarray:
    """One MinHash signature (``num_perm`` values) for each text. An empty text gets the maximum value."""
    rng = np.random.default_rng(seed)
    salts = rng.integers(0, 2**63, num_perm, dtype=np.uint64)
    sigs = np.full((len(texts), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i, t in enumerate(texts):
            hs = np.fromiter((_h64(x) for x in shingles(t)), dtype=np.uint64)
            if hs.size:
                sigs[i] = _mix(hs[:, None] ^ salts[None, :]).min(axis=0)
    return sigs


def group_duplicates(df: pd.DataFrame, threshold: float = 0.8, num_perm: int = 64, bands: int = 16,
                     seed: int = 1) -> tuple[pd.DataFrame, DedupReport]:
    n = len(df)
    uf = UnionFind(n)
    norm = df["text"].map(normalise).tolist()
    first: dict[str, int] = {}
    exact = 0
    for i, t in enumerate(norm):
        if t in first:
            uf.union(first[t], i)
            exact += 1
        else:
            first[t] = i
    sigs = minhash_signatures(df["text"].tolist(), num_perm, seed)
    rows = num_perm // bands
    links = 0
    for b in range(bands):
        buckets: dict[bytes, list[int]] = {}
        for i in range(n):
            buckets.setdefault(sigs[i, b * rows:(b + 1) * rows].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for j in members[1:]:
                i0 = members[0]
                if uf.find(i0) != uf.find(j) and (sigs[i0] == sigs[j]).mean() >= threshold:
                    uf.union(i0, j)
                    links += 1
    out = df.copy()
    out["group"] = [f"g{uf.find(i)}" for i in range(n)]
    conflicts = int((out.groupby("group")["label"].nunique() > 1).sum())
    return out, DedupReport(n, exact, links, out["group"].nunique(), conflicts)


def resolve(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one post for each group. A group with a label conflict keeps its majority label;
    a tie drops the group, because its true label is unknown."""
    keep = []
    for _, g in df.groupby("group", sort=False):
        counts = g["label"].value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            continue
        keep.append(g[g["label"] == counts.index[0]].iloc[0])
    return pd.DataFrame(keep).reset_index(drop=True)
