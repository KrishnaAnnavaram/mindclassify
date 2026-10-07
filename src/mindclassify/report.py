"""The evaluation report and the model card. Every number is written as computed."""

from __future__ import annotations

from pathlib import Path

from . import DISCLAIMER, SAFETY_LABEL
from .prompts import TEMPLATE_VERSION


def _f(x) -> str:
    return f"{x:.3f}"


def markdown(results: list[dict], prepared) -> str:
    lines = ["# mindclassify evaluation report", "", f"> {DISCLAIMER}", "",
             f"Data: {'SYNTHETIC posts (not real people)' if prepared.synthetic else 'local corpus'}. "
             f"{prepared.load_report.rows_out} rows loaded, {prepared.removed_by_dedup} removed as duplicates "
             f"({prepared.dedup_report.label_conflict_groups} duplicate groups had two labels). Test split: "
             f"{len(prepared.split.test)} posts. Prompt template `{TEMPLATE_VERSION}`.", "",
             "| Model | Macro-F1 | 95% interval | Balanced acc. | Accuracy | ECE | Suicidal routing recall |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lo, hi = r["macro_f1_ci95"]
        rec = "n/a"
        if "safety" in r:
            rec = _f(r["safety"]["recall"])
            if "with_phrases" in r["safety"]:
                rec += f" (with phrases {_f(r['safety']['with_phrases']['recall'])})"
        lines.append(f"| `{r['model']}` | {_f(r['macro_f1'])} | [{_f(lo)}, {_f(hi)}] | {_f(r['balanced_accuracy'])} "
                     f"| {_f(r['accuracy'])} | {_f(r['ece'])} | {rec} |")
    for r in results:
        lines += ["", f"## Per-class results: `{r['model']}`", "", "| Label | Precision | Recall | F1 | Support |",
                  "|---|---|---|---|---|"]
        for lab, v in r["per_class"].items():
            lines.append(f"| {lab} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['f1'])} | {v['support']} |")
        c = r.get("calibration")
        if c:
            lines.append(f"\nCalibration on the validation split: temperature {c['temperature']:.3f}, ECE "
                         f"{c['val_ece_before']:.3f} -> {c['val_ece_after']:.3f}.")
        if "safety" in r:
            sft = r["safety"]
            lines.append(f"\n{SAFETY_LABEL} routing rule: P >= {sft['threshold']:.3f} -> human review. "
                         f"Recall {_f(sft['recall'])}, precision {_f(sft['precision'])}, "
                         f"routed share {_f(sft['routed_share'])}.")
            if "with_phrases" in sft:
                w = sft["with_phrases"]
                lines.append(f"With the crisis-phrase rule added: recall {_f(w['recall'])}, precision "
                             f"{_f(w['precision'])}, routed share {_f(w['routed_share'])}.")
        lines += ["", "| Slice | Value | n | Macro-F1 |", "|---|---|---|---|"]
        lines += [f"| {x['slice']} | {x['value']} | {x['n']} | {_f(x['macro_f1'])} |" for x in r["slices"]]
    return "\n".join(lines) + "\n"


def model_card(meta: dict, result: dict | None) -> str:
    perf = "Not evaluated." if result is None else (
        f"Test macro-F1 {_f(result['macro_f1'])} (95% interval {_f(result['macro_f1_ci95'][0])} to "
        f"{_f(result['macro_f1_ci95'][1])}), {result['n']} posts.")
    return "\n".join([
        f"# Model card: mindclassify `{meta['type']}`", "", f"> {DISCLAIMER}", "",
        "## Intended use",
        "Research on the language of public, self-disclosed mental-health posts. Not for diagnosis, screening, "
        "triage or any decision about a person.", "",
        "## Model", f"Type `{meta['type']}`, labels {meta['labels']}, temperature {meta['temperature']:.3f}, "
        f"prompt template `{meta['template_version']}`, safety threshold {meta.get('safety_threshold')}.", "",
        "## Performance", perf, "",
        "## Limits and bias",
        "- The labels come from the subreddit or the source dataset, not from a clinician.",
        "- The corpus over-represents some platforms, languages and demographic groups.",
        "- A post with suicidal content must go to a human reviewer and crisis resources, whatever the top label.",
        "",
    ])


def write(text: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
