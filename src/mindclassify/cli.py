"""Command line: ``mindclassify synth | prepare | train | evaluate | predict | train-llm | prompt | config``.

RESEARCH USE ONLY. No command gives a diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import DISCLAIMER, SAFETY_LABEL
from .config import Settings
from .evaluate import calibrate_and_route, evaluate
from .models import LOCAL, build, load, save
from .pipeline import labels_present, prepare
from .prompts import TEMPLATE, TEMPLATE_VERSION
from .report import markdown, model_card, write
from .safety import phrase_flags
from .splits import label_table
from .synthetic import SynthSpec, generate


def _settings(args) -> Settings:
    keys = ("data_path", "out_dir", "seed", "safety_recall", "base_model", "epochs", "lora_r", "lora_alpha")
    return Settings.from_env().merge(**{k: getattr(args, k, None) for k in keys})


def cmd_synth(args) -> int:
    df = generate(SynthSpec(n=args.n, seed=args.seed))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} synthetic posts to {args.out}")
    return 0


def cmd_prepare(args) -> int:
    s = _settings(args)
    p = prepare(s, synthetic=args.synthetic)
    print(f"load: {vars(p.load_report)}")
    print(f"dedup: {vars(p.dedup_report)}; removed {p.removed_by_dedup} posts")
    print(label_table(p.split, labels_present(p)).to_string())
    path = p.split.save(Path(s.out_dir) / "splits.json")
    print(f"split IDs saved to {path}")
    return 0


def cmd_train(args) -> int:
    s = _settings(args)
    p = prepare(s, synthetic=args.synthetic)
    labels = labels_present(p)
    results = []
    for name in args.models.split(","):
        model = build(name, labels, s.seed).fit(p.split.train["text"].tolist(), p.split.train["label"].tolist())
        c = calibrate_and_route(model, p.split.val, s.safety_recall)
        res = evaluate(model, p.split.test, c.get("safety_threshold"), seed=s.seed)
        res["calibration"] = c
        results.append(res)
        out = save(model, Path(args.out) / name, {"safety_threshold": c.get("safety_threshold"),
                                                  "calibration": c, "seed": s.seed})
        meta = json.loads((out / "model.json").read_text(encoding="utf-8"))
        write(model_card(meta, res), out / "MODEL_CARD.md")
        print(f"{name}: test macro-F1 {res['macro_f1']:.3f}, saved to {out}")
    path = write(markdown(results, p), Path(args.out) / "report.md")
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_evaluate(args) -> int:
    s = _settings(args)
    p = prepare(s, synthetic=args.synthetic)
    model, meta = load(args.model_dir, s)
    res = evaluate(model, p.split.test, meta.get("safety_threshold"), seed=s.seed)
    res["calibration"] = meta.get("calibration")
    path = write(markdown([res], p), Path(args.out or s.out_dir) / "report.md")
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_predict(args) -> int:
    s = _settings(args)
    model, meta = load(args.model_dir, s)
    probs = model.predict_proba([args.text])[0]
    ranked = sorted(zip(model.labels, probs), key=lambda kv: -kv[1])
    out = {"top_label": ranked[0][0], "probabilities": {k: round(float(v), 4) for k, v in ranked},
           "disclaimer": DISCLAIMER}
    t = meta.get("safety_threshold")
    if t is not None and SAFETY_LABEL in model.labels:
        p_s = float(probs[model.labels.index(SAFETY_LABEL)])
        phrase = bool(phrase_flags([args.text])[0])
        out["route_to_human"] = bool(p_s >= t or phrase)
        if out["route_to_human"]:
            why = f"P({SAFETY_LABEL}) = {p_s:.3f} (threshold {t:.3f})" + (", crisis phrase found" if phrase else "")
            out["routing_note"] = f"{why}. Send this post to a trained human reviewer and to crisis resources now."
    print(json.dumps(out, indent=2))
    return 0


def cmd_train_llm(args) -> int:  # pragma: no cover - needs a GPU and the LLM extra
    from .models.qlora import train_qlora

    s = _settings(args)
    p = prepare(s, synthetic=args.synthetic)
    for seed in [int(x) for x in args.seeds.split(",")]:
        out = train_qlora(s, p.split.train, p.split.val, Path(args.out) / f"seed{seed}", seed=seed)
        model, meta = load(out, s)
        c = calibrate_and_route(model, p.split.val, s.safety_recall)
        meta.update({"temperature": c["temperature"], "safety_threshold": c.get("safety_threshold"), "calibration": c})
        (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"seed {seed}: adapter in {out}")
    return 0


def cmd_prompt(args) -> int:
    print(f"template {TEMPLATE_VERSION}:\n{TEMPLATE}")
    return 0


def cmd_config(args) -> int:
    for k, v in _settings(args).public().items():
        print(f"{k:>20} = {v}")
    return 0


def _data(p) -> None:
    p.add_argument("--synthetic", action="store_true", help="use synthetic posts")
    p.add_argument("--data-path", help="the corpus CSV (MINDCLASSIFY_DATA)")
    p.add_argument("--seed", type=int)
    p.add_argument("--safety-recall", type=float)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mindclassify", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("synth", help="write a synthetic corpus CSV")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("prepare", help="load, clean, de-duplicate and split; save the split IDs")
    _data(p)
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("train", help="train local models, calibrate on validation, evaluate on test")
    _data(p)
    p.add_argument("--models", default="majority,tfidf", help=f"comma list of {sorted(LOCAL)}")
    p.add_argument("--out", default="models")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("evaluate", help="evaluate a saved model folder on the test split")
    _data(p)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("predict", help="class probabilities and the safety routing for one text")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--text", required=True)
    p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("train-llm", help="QLoRA fine-tuning with a completion-only loss (GPU, extra llm)")
    _data(p)
    p.add_argument("--base-model")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lora-r", type=int)
    p.add_argument("--lora-alpha", type=int)
    p.add_argument("--seeds", default="42")
    p.add_argument("--out", default="models/qlora")
    p.set_defaults(fn=cmd_train_llm)
    p = sub.add_parser("prompt", help="show the one prompt template")
    p.set_defaults(fn=cmd_prompt)
    p = sub.add_parser("config", help="show the settings (the token shows only as set or not set)")
    p.set_defaults(fn=cmd_config)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
