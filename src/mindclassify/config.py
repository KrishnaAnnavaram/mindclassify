"""Settings from environment variables. Tokens are read from the environment and never printed."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace


@dataclass(frozen=True)
class Settings:
    data_path: str = "data/Combined Data.csv"
    out_dir: str = "out"
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    near_dup_threshold: float = 0.8
    safety_recall: float = 0.9  # target recall of the Suicidal routing rule on the validation split
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 2
    learning_rate: float = 2e-4
    max_length: int = 384
    hf_token: str = field(default="", repr=False)

    ENV = {
        "data_path": "MINDCLASSIFY_DATA", "out_dir": "MINDCLASSIFY_OUT_DIR", "seed": "MINDCLASSIFY_SEED",
        "test_size": "MINDCLASSIFY_TEST_SIZE", "val_size": "MINDCLASSIFY_VAL_SIZE",
        "near_dup_threshold": "MINDCLASSIFY_NEAR_DUP_THRESHOLD", "safety_recall": "MINDCLASSIFY_SAFETY_RECALL",
        "base_model": "MINDCLASSIFY_BASE_MODEL", "lora_r": "MINDCLASSIFY_LORA_R",
        "lora_alpha": "MINDCLASSIFY_LORA_ALPHA", "epochs": "MINDCLASSIFY_EPOCHS",
        "learning_rate": "MINDCLASSIFY_LEARNING_RATE", "max_length": "MINDCLASSIFY_MAX_LENGTH",
        "hf_token": "HF_TOKEN",
    }

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        types = {f.name: f.type for f in fields(cls)}
        values = {}
        for name, var in cls.ENV.items():
            raw = env.get(var)
            if raw:
                t = types[name]
                values[name] = int(raw) if t in ("int", int) else float(raw) if t in ("float", float) else raw
        return cls(**values).validate()

    def merge(self, **kw) -> "Settings":
        return replace(self, **{k: v for k, v in kw.items() if v is not None}).validate()

    def validate(self) -> "Settings":
        if not (0 < self.test_size < 0.5 and 0 < self.val_size < 0.5):
            raise ValueError("test_size and val_size must be between 0 and 0.5")
        if not 0 < self.safety_recall <= 1:
            raise ValueError("safety_recall must be in (0, 1]")
        return self

    def public(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "hf_token"}
        out["hf_token"] = "set" if self.hf_token else "not set"
        return out
