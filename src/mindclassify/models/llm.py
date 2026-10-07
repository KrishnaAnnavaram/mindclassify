"""LLM classification by LABEL SCORING (extra ``llm``). torch and transformers are imported lazily.

For each post, the scorer builds the shared prompt and adds each label completion. The score of a
label is the sum of the log-probabilities of its tokens. A softmax over the label scores gives
class probabilities. The model never generates free text, so no output parsing is necessary.
"""

from __future__ import annotations

import numpy as np

from .. import LABELS
from ..prompts import build_prompt, completion
from .base import Classifier


def score_labels(prompt_ids: list[int], label_ids: list[list[int]], token_logprobs) -> np.ndarray:
    """Pure helper: ``token_logprobs(sequence, start)`` returns the log-probabilities of
    ``sequence[start:]``. The result is one summed score for each label."""
    out = np.empty(len(label_ids))
    for j, lab in enumerate(label_ids):
        out[j] = float(np.sum(token_logprobs(prompt_ids + lab, len(prompt_ids))))
    return out


class LLMLabelScorer(Classifier):  # pragma: no cover - needs a GPU model download
    name = "llm"

    def __init__(self, base_model: str, adapter: str | None = None, token: str = "", labels=LABELS,
                 load_4bit: bool = True, batch_size: int = 8):
        super().__init__(labels)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError('the LLM extra is not installed: pip install -e ".[llm]"') from exc
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(base_model, token=token or None)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16) if load_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant, device_map="auto",
                                                          token=token or None)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.label_ids = [self.tok(completion(l), add_special_tokens=False)["input_ids"] for l in self.labels]
        self.batch_size = batch_size
        self.name = "llm-qlora" if adapter else "llm-zero-shot"

    def _batch_logprobs(self, seqs: list[list[int]], starts: list[int]) -> list[float]:
        torch = self.torch
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        with torch.no_grad():
            logits = self.model(input_ids=ids.to(self.model.device), attention_mask=mask.to(self.model.device)).logits
        logp = torch.log_softmax(logits.float(), dim=-1).cpu()
        out = []
        for i, (s, st) in enumerate(zip(seqs, starts)):
            pos = torch.arange(st, len(s))
            out.append(float(logp[i, pos - 1, torch.tensor(s[st:])].sum()))
        return out

    def log_scores(self, texts) -> np.ndarray:
        rows = []
        for t in texts:
            p = self.tok(build_prompt(t, self.labels))["input_ids"]
            seqs = [p + lab for lab in self.label_ids]
            rows.append(self._batch_logprobs(seqs, [len(p)] * len(seqs)))
        return np.array(rows)
