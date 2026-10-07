"""A crisis-phrase rule that works next to the model threshold.

A model can miss an explicit phrase. The routing rule is therefore: route the post to a human if
P(Suicidal) >= threshold OR the text contains a crisis phrase. The phrase list favours recall.
It is a starting list for review by a clinical safety team, not a complete one.
"""

from __future__ import annotations

import re

import numpy as np

CRISIS_PHRASES = (
    r"\bkill (?:my ?self|me)\b", r"\bsuicid\w*", r"\bend (?:it|it all|my life)\b", r"\bwant(?:ed)? to die\b",
    r"\bno way out\b", r"\bnot (?:want to )?be here\b", r"\bbetter (?:off )?without me\b", r"\bgoodbye note\b",
    r"\bself[- ]harm\w*", r"\boverdose\b", r"\bfinal plan\b",
)
_RX = re.compile("|".join(CRISIS_PHRASES), re.I)


def phrase_flags(texts) -> np.ndarray:
    return np.array([bool(_RX.search(t)) for t in texts], dtype=bool)


def route(p_safety, texts, threshold: float) -> np.ndarray:
    return (np.asarray(p_safety) >= threshold) | phrase_flags(texts)
