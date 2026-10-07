import json
from pathlib import Path

import numpy as np
import pytest

from mindclassify import DISCLAIMER
from mindclassify.cli import main
from mindclassify.config import Settings
from mindclassify.evaluate import (calibrate_and_route, ece, evaluate, fit_temperature, metrics, safety_metrics,
                                   safety_threshold)
from mindclassify.models import build, load, save
from mindclassify.models.base import softmax
from mindclassify.pipeline import labels_present
from mindclassify.safety import phrase_flags

REPO = Path(__file__).resolve().parents[1]
SECRET = "hf_" + "Q" * 34


@pytest.fixture(scope="module")
def trained(prepared, settings):
    labels = labels_present(prepared)
    tr = prepared.split.train
    out = {}
    for name in ("majority", "tfidf"):
        m = build(name, labels, settings.seed).fit(tr["text"].tolist(), tr["label"].tolist())
        c = calibrate_and_route(m, prepared.split.val, 0.9)
        out[name] = (m, c, evaluate(m, prepared.split.test, c["safety_threshold"], n_boot=200))
    return out


def test_baseline_ladder_and_macro_f1(trained):
    """Problem 5: macro-F1 is the primary metric, and every model is compared with the majority floor."""
    maj, tfidf = trained["majority"][2], trained["tfidf"][2]
    assert maj["macro_f1"] < 0.1 and tfidf["macro_f1"] > 0.7
    lo, hi = tfidf["macro_f1_ci95"]
    assert lo <= tfidf["macro_f1"] <= hi
    assert set(tfidf["per_class"]) == set(trained["tfidf"][0].labels)
    assert {s["slice"] for s in tfidf["slices"]} == {"source", "length"}


def test_temperature_scaling_fixes_overconfidence():
    """Problem 6: probabilities are calibrated on validation data."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(0, 1, (2000, 3))
    logits[np.arange(2000), y] += 1.0
    hot = logits * 5  # over-confident scores
    t = fit_temperature(hot, y)
    assert t > 2
    assert ece(softmax(hot, t), y) < ece(softmax(hot, 1.0), y)


def test_metrics_known_values():
    labels = ["A", "B"]
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    m = metrics(labels, ["A", "B", "B", "B"], probs, n_boot=50)
    assert m["accuracy"] == 0.75 and m["confusion"] == [[1, 0], [1, 2]]
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_safety_threshold_reaches_the_target_recall():
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1, 0.05])
    is_s = np.array([True, True, True, True, False, False])
    t = safety_threshold(p, is_s, 0.75)
    assert t == pytest.approx(0.3)
    m = safety_metrics(p, is_s, t, ["x"] * 4 + ["I want to end it", "x"])
    assert m["recall"] == 0.75 and m["with_phrases"]["routed_share"] == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        safety_threshold(p, np.zeros(6, bool), 0.9)


def test_crisis_phrases():
    flags = phrase_flags(["I want to end it all", "end of the season", "thinking about suicide", "fine"])
    assert flags.tolist() == [True, False, True, False]


def test_save_and_load(trained, tmp_path):
    m, c, _ = trained["tfidf"]
    save(m, tmp_path / "t", {"safety_threshold": c["safety_threshold"], "calibration": c})
    again, meta = load(tmp_path / "t")
    texts = ["empty and hopeless tonight", "football with my cousin"]
    assert np.allclose(again.predict_proba(texts), m.predict_proba(texts))
    meta_path = tmp_path / "t" / "model.json"
    bad = json.loads(meta_path.read_text())
    bad["template_version"] = "old"
    meta_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load(tmp_path / "t")


def test_token_is_never_printed(monkeypatch, capsys):
    """Problem 9: the token comes from the environment and is never shown."""
    monkeypatch.setenv("HF_TOKEN", SECRET)
    assert SECRET not in repr(Settings.from_env())
    assert main(["config"]) == 0
    out = capsys.readouterr().out
    assert SECRET not in out and "hf_token = set" in out


def test_no_notebook_with_outputs_and_no_secret_in_sources():
    for nb in REPO.rglob("*.ipynb"):
        if ".venv" not in nb.parts:
            cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
            assert not any(c.get("outputs") for c in cells), nb
    for py in (REPO / "src").rglob("*.py"):
        assert "hf_" + "" not in py.read_text(encoding="utf-8").replace("hf_token", "")


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "posts.csv"
    assert main(["synth", "--out", str(data), "--n", "1500", "--seed", "5"]) == 0
    assert main(["prepare", "--data-path", str(data), "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "splits.json").exists()
    assert main(["train", "--data-path", str(data), "--out", str(tmp_path / "m")]) == 0
    report = (tmp_path / "m" / "report.md").read_text(encoding="utf-8")
    assert "| `tfidf` |" in report and DISCLAIMER in report
    assert "RESEARCH USE ONLY" in (tmp_path / "m" / "tfidf" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert main(["evaluate", "--data-path", str(data), "--model-dir", str(tmp_path / "m" / "tfidf"),
                 "--out", str(tmp_path / "e")]) == 0
    capsys.readouterr()
    assert main(["predict", "--model-dir", str(tmp_path / "m" / "tfidf"), "--text", "I want to end it"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["route_to_human"] is True and "crisis phrase" in out["routing_note"]
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert main(["prompt"]) == 0 and "Label:" in capsys.readouterr().out
    assert main(["prepare", "--data-path", str(tmp_path / "none.csv")]) == 2
