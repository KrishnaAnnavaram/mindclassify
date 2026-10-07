"""The ONE prompt template. Training, evaluation and inference all call ``build_prompt``.

The completion is a space plus the label text. The model never generates free text: the LLM
classifier scores each label completion and normalises the scores over the label set.
"""

from __future__ import annotations

from . import LABELS

TEMPLATE_VERSION = "classify-v1"
TEMPLATE = (
    "Classify the mental-health category of the post. Answer with exactly one label from this list: "
    "{labels}.\n\nPost: {text}\nLabel:"
)


def build_prompt(text: str, labels=LABELS) -> str:
    return TEMPLATE.format(labels=", ".join(labels), text=" ".join(text.split()))


def completion(label: str) -> str:
    return f" {label}"


def training_example(text: str, label: str, labels=LABELS) -> tuple[str, str]:
    """(prompt, completion). The loss covers only the completion."""
    if label not in labels:
        raise ValueError(f"unknown label {label!r}")
    return build_prompt(text, labels), completion(label)


def completion_only_labels(prompt_ids: list[int], completion_ids: list[int], eos_id: int | None,
                           max_length: int, keep_end: int = 4, ignore_index: int = -100) -> tuple[list[int], list[int]]:
    """Token IDs and training labels. Prompt tokens get ``ignore_index``, so only the label is learned.

    If the sequence is too long, the END OF THE POST is cut. The instruction at the start, the last
    ``keep_end`` prompt tokens (the "Label:" marker) and the whole completion always stay.
    """
    tail = completion_ids + ([eos_id] if eos_id is not None else [])
    room = max_length - len(tail)
    if room <= keep_end:
        raise ValueError("max_length is too small for the completion")
    p = prompt_ids if len(prompt_ids) <= room else prompt_ids[:room - keep_end] + prompt_ids[-keep_end:]
    return p + tail, [ignore_index] * len(p) + tail
