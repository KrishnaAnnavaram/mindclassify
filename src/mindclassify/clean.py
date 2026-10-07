"""Minimal, signal-safe text cleaning.

The cleaner replaces URLs and user mentions with placeholders, removes HTML tags and entities, and
collapses white space. It KEEPS negations, pronouns and word order, because they carry the signal.
Slang expansion is off by default. When it is on, it changes only whole tokens and keeps the case
style of the token. Each step is a switch, so an ablation can measure it.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
MENTION = re.compile(r"(?<!\w)@\w+")
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
SLANG = {"u": "you", "ur": "your", "idk": "I do not know", "imo": "in my opinion", "rn": "right now",
         "tbh": "to be honest", "bc": "because", "cant": "can not", "dont": "do not", "wont": "will not",
         "im": "I am", "ive": "I have", "pls": "please", "thx": "thanks"}
_WORD = re.compile(r"\b[\w']+\b")


@dataclass(frozen=True)
class CleanConfig:
    urls: bool = True
    mentions: bool = True
    html: bool = True
    lowercase: bool = False
    expand_slang: bool = False


def expand_slang(text: str) -> str:
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        out = SLANG.get(tok.lower())
        if out is None:
            return tok
        return out[0].upper() + out[1:] if tok[0].isupper() else out

    return _WORD.sub(repl, text)


def clean(text: str, cfg: CleanConfig = CleanConfig()) -> str:
    t = text
    if cfg.html:
        t = TAG.sub(" ", html.unescape(t))
    if cfg.urls:
        t = URL.sub(" <url> ", t)
    if cfg.mentions:
        t = MENTION.sub(" <user> ", t)
    if cfg.expand_slang:
        t = expand_slang(t)
    if cfg.lowercase:
        t = t.lower()
    return SPACE.sub(" ", t).strip()
