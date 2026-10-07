import numpy as np
import pandas as pd
import pytest

from mindclassify import LABELS
from mindclassify.dedup import group_duplicates, normalise, resolve
from mindclassify.models.llm import score_labels
from mindclassify.models.qlora import encode
from mindclassify.pipeline import labels_present
from mindclassify.prompts import build_prompt, completion, completion_only_labels, training_example
from mindclassify.splits import LeakageError, Split, make_split


def frame(texts, labels):
    return pd.DataFrame({"id": [f"i{k}" for k in range(len(texts))], "text": texts, "label": labels,
                         "source": "t"})


def test_exact_and_near_duplicates_share_a_group():
    base = "today I feel empty and tired again at night with my family and friends"
    df = frame([base, base.upper() + "!", base + " really", "a completely different post about football games"],
               ["Depression"] * 3 + ["Normal"])
    g, rep = group_duplicates(df, threshold=0.6)
    assert g["group"].iloc[0] == g["group"].iloc[1] == g["group"].iloc[2] != g["group"].iloc[3]
    assert rep.exact_duplicates == 1 and rep.groups == 2


def test_resolve_majority_and_ties():
    df = pd.DataFrame({"id": list("abcde"), "text": list("abcde"), "group": ["g1", "g1", "g1", "g2", "g2"],
                       "label": ["Stress", "Stress", "Anxiety", "Normal", "Anxiety"]})
    out = resolve(df)
    assert out["label"].tolist() == ["Stress"]  # g2 is a tie and is dropped


def test_split_has_no_duplicate_across_sides(prepared, tmp_path):
    """Problem 3: no duplicate group and no normalised text is in two splits."""
    sp = prepared.split
    sp.check()
    assert prepared.removed_by_dedup > 0
    n = len(prepared.data)
    assert len(sp.train) + len(sp.val) + len(sp.test) == n
    assert abs(len(sp.test) / n - 0.15) < 0.04
    again = Split.from_ids(prepared.data, sp.save(tmp_path / "s.json"))
    assert again.test["id"].tolist() == sp.test["id"].tolist()


def test_leakage_check_catches_a_copy(prepared):
    sp = prepared.split
    bad = Split(sp.train, sp.val, pd.concat([sp.test, sp.train.head(1)]))
    with pytest.raises(LeakageError):
        bad.check()


def test_suicidal_class_is_kept(prepared):
    """Problem 4: the safety-critical class is in every split."""
    assert "Suicidal" in labels_present(prepared)
    for part in (prepared.split.train, prepared.split.val, prepared.split.test):
        assert (part["label"] == "Suicidal").sum() > 10


def test_stratified_split_keeps_label_shares():
    rng = np.random.default_rng(0)
    labels = rng.choice(["A", "B"], size=1000, p=[0.8, 0.2])
    df = frame([f"post {i}" for i in range(1000)], labels)
    sp = make_split(df, 0.2, 0.2, seed=1)
    for part in (sp.train, sp.val, sp.test):
        assert abs((part["label"] == "B").mean() - 0.2) < 0.05


def test_one_prompt_for_training_and_inference():
    """Problem 1: the training prompt and the inference prompt are the same function and text."""
    import mindclassify.models.llm as llm
    import mindclassify.models.qlora as qlora
    import mindclassify.prompts as prompts

    assert llm.build_prompt is prompts.build_prompt and qlora.build_prompt is prompts.build_prompt
    p, c = training_example("I  feel\nfine", "Normal")
    assert p == build_prompt("I feel fine") and c == " Normal"
    assert p.endswith("Post: I feel fine\nLabel:")
    with pytest.raises(ValueError):
        training_example("x", "Happy")


def test_completion_only_loss_and_truncation():
    """Problem 7: only the label tokens carry a loss. The instruction and the marker survive a cut."""
    prompt = list(range(100, 130))
    ids, labels = completion_only_labels(prompt, [7, 8], eos_id=2, max_length=50)
    assert ids == prompt + [7, 8, 2] and labels == [-100] * 30 + [7, 8, 2]
    ids, labels = completion_only_labels(prompt, [7, 8], eos_id=2, max_length=20, keep_end=4)
    assert len(ids) == 20 and ids[:13] == prompt[:13] and ids[13:17] == prompt[-4:]
    assert labels.count(-100) == 17
    with pytest.raises(ValueError):
        completion_only_labels(prompt, [7, 8], 2, max_length=6, keep_end=4)


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1 + (ord(ch) % 50) for ch in text][:40]}


def test_encode_masks_each_example():
    rows = encode(FakeTokenizer(), ["I feel ok"], ["Normal"], max_length=60)
    ids, lab = rows[0]["input_ids"], rows[0]["labels"]
    n_label = len(FakeTokenizer()(completion("Normal"))["input_ids"]) + 1
    assert len(ids) == len(lab) and sum(x != -100 for x in lab) == n_label


def test_label_scoring_gives_a_distribution_without_generation():
    """Problems 6 and 10: probabilities over the label set, no text parsing."""
    label_ids = [[10], [20, 21], [30]]

    def token_logprobs(seq, start):  # a fake LM that likes token 20 and 21
        return np.array([-0.1 if t in (20, 21) else -3.0 for t in seq[start:]])

    scores = score_labels([1, 2, 3], label_ids, token_logprobs)
    assert scores.argmax() == 1 and scores[1] == pytest.approx(-0.2)
    from mindclassify.models.base import softmax

    p = softmax(scores[None, :])[0]
    assert p.sum() == pytest.approx(1.0) and p.argmax() == 1
