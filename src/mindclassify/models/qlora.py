"""QLoRA fine-tuning with a completion-only loss (extra ``llm``). Not run in CI.

- 4-bit NF4 base model, LoRA on all linear layers.
- Each example is ``build_prompt(text)`` + ``" " + label`` + EOS. Prompt tokens get the label -100,
  so the loss covers only the label tokens (``prompts.completion_only_labels``).
- Evaluation on the validation split after each epoch, early stopping on the validation loss,
  and the best checkpoint is kept.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import LABELS
from ..prompts import TEMPLATE_VERSION, completion, build_prompt, completion_only_labels


def encode(tok, texts, labels, max_length: int, label_set=LABELS) -> list[dict]:
    rows = []
    for t, y in zip(texts, labels):
        p = tok(build_prompt(t, label_set))["input_ids"]
        c = tok(completion(y), add_special_tokens=False)["input_ids"]
        ids, lab = completion_only_labels(p, c, tok.eos_token_id, max_length)
        rows.append({"input_ids": ids, "labels": lab})
    return rows


def train_qlora(settings, train_df, val_df, out_dir: str | Path, seed: int | None = None) -> Path:  # pragma: no cover
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback,
                                  Trainer, TrainingArguments)
    except ImportError as exc:
        raise RuntimeError('the LLM extra is not installed: pip install -e ".[llm]"') from exc
    seed = settings.seed if seed is None else seed
    torch.manual_seed(seed)
    token = settings.hf_token or None
    tok = AutoTokenizer.from_pretrained(settings.base_model, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(settings.base_model, quantization_config=quant, device_map="auto",
                                                 token=token)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=settings.lora_r, lora_alpha=settings.lora_alpha, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    train = encode(tok, train_df["text"], train_df["label"], settings.max_length)
    val = encode(tok, val_df["text"], val_df["label"], settings.max_length)

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id)
        lab = torch.full((len(batch), width), -100)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], lab[i, :n], att[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    out = Path(out_dir)
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=settings.epochs, learning_rate=settings.learning_rate,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, per_device_eval_batch_size=8,
        lr_scheduler_type="cosine", warmup_ratio=0.03, optim="paged_adamw_8bit", bf16=True,
        gradient_checkpointing=True, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2, seed=seed,
        report_to="none", logging_steps=20, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.train()
    model.save_pretrained(out / "adapter")
    (out / "model.json").write_text(json.dumps({
        "type": "llm", "base_model": settings.base_model, "adapter": "adapter", "labels": list(LABELS),
        "template_version": TEMPLATE_VERSION, "seed": seed, "lora_r": settings.lora_r,
        "lora_alpha": settings.lora_alpha, "temperature": 1.0, "safety_threshold": None,
    }, indent=2), encoding="utf-8")
    return out
