import pandas as pd
import pytest

from mindclassify import LABELS
from mindclassify.clean import CleanConfig, clean, expand_slang
from mindclassify.data import SchemaError, load_csv, load_frame, normalise_label


def test_label_normalisation():
    assert normalise_label("Bi-Polar") == "Bipolar"
    assert normalise_label(" suicidal ") == "Suicidal"
    assert normalise_label("Personality Disorder") == "Personality disorder"
    assert normalise_label("happy") is None and normalise_label(None) is None


def test_load_frame_drops_bad_rows_and_keeps_suicidal():
    raw = pd.DataFrame({"statement": ["a", "", None, "b", "c"],
                        "status": ["Suicidal", "Normal", "Normal", "Unknown", "Bi-Polar"]})
    df, rep = load_frame(raw, "x")
    assert df["label"].tolist() == ["Suicidal", "Bipolar"]
    assert (rep.empty_text, rep.unknown_label) == (2, 1) and rep.unknown_values == {"Unknown": 1}
    assert set(df["source"]) == {"x"}
    with pytest.raises(SchemaError):
        load_frame(pd.DataFrame({"text": ["a"]}), "x")


def test_load_csv(tmp_path):
    p = tmp_path / "Combined Data.csv"
    pd.DataFrame({"Unnamed: 0": [0], "statement": ["hello"], "status": ["Anxiety"]}).to_csv(p, index=False)
    df, _ = load_csv(p)
    assert df.loc[0, "label"] == "Anxiety" and df.loc[0, "id"] == "Combined_Data-0"
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "none.csv")


def test_cleaner_keeps_negations_and_word_order():
    """Problem 8: negations and wording carry the signal and must stay."""
    text = "I do NOT feel ok, never again &amp; no hope <b>today</b> @friend https://x.y/z"
    out = clean(text)
    assert out == "I do NOT feel ok, never again & no hope today <user> <url>"
    assert "not" in clean(text, CleanConfig(lowercase=True)).split()


def test_slang_expansion_is_whole_token_and_case_aware():
    assert expand_slang("time stats u U idk") == "time stats you You I do not know"
    assert clean("u ok", CleanConfig(expand_slang=True)) == "you ok"
    assert clean("u ok") == "u ok"  # off by default


def test_label_order_is_fixed():
    assert LABELS[2] == "Suicidal" and len(LABELS) == 7
