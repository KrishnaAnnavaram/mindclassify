"""Model registry and the model folder (``model.json`` + ``model.joblib`` or an adapter)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ..prompts import TEMPLATE_VERSION
from .base import Classifier, MajorityClassifier, softmax
from .tfidf import TfidfClassifier

LOCAL = {"majority": MajorityClassifier, "tfidf": TfidfClassifier}


def build(name: str, labels, seed: int = 42) -> Classifier:
    if name not in LOCAL:
        raise ValueError(f"unknown model {name!r}. Local models: {sorted(LOCAL)}. Use 'train-llm' for QLoRA")
    return LOCAL[name](labels=labels) if name == "majority" else LOCAL[name](labels=labels, seed=seed)


def save(model: Classifier, out_dir: str | Path, meta: dict) -> Path:
    """Write the model folder. joblib uses pickle: load only model folders that you made."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    (out / "model.json").write_text(json.dumps({
        "type": model.name, "labels": model.labels, "temperature": model.temperature,
        "template_version": TEMPLATE_VERSION, **meta}, indent=2), encoding="utf-8")
    return out


def load(model_dir: str | Path, settings=None) -> tuple[Classifier, dict]:
    d = Path(model_dir)
    meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
    if meta.get("template_version") != TEMPLATE_VERSION:
        raise ValueError(f"model prompt template {meta.get('template_version')} != {TEMPLATE_VERSION}")
    if meta["type"] == "llm":  # pragma: no cover - needs the LLM extra
        from .llm import LLMLabelScorer

        token = settings.hf_token if settings else ""
        model = LLMLabelScorer(meta["base_model"], str(d / meta["adapter"]) if meta.get("adapter") else None,
                               token, meta["labels"])
        model.temperature = meta.get("temperature", 1.0)
        return model, meta
    model = joblib.load(d / "model.joblib")
    model.temperature = meta.get("temperature", 1.0)
    return model, meta


__all__ = ["Classifier", "LOCAL", "build", "load", "save", "softmax"]
