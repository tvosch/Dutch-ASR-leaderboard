"""Tests for the asr_nl package (no GPU, network or model downloads needed)."""

import numpy as np
import pytest

import asr_nl
from asr_nl.audio import to_mono
from asr_nl.cli.rescore import rescore
from asr_nl.datasets import DATASETS
from asr_nl.evaluation import score_samples
from asr_nl.text import NORMALIZER_VERSION, normalize_text


def test_version():
    assert asr_nl.__version__


def test_every_dataset_declares_language():
    for name, cfg in DATASETS.items():
        assert cfg["language"] in ("nl", "en"), name


@pytest.mark.parametrize("raw, expected", [
    ("Zeker 20% van al het water.", "zeker twintig van al het water"),
    ("de auto's", "de autos"),
    ("ING en Mr. Holmes", "ing en mr holmes"),
    ("Noord-Holland, uh, ja", "noordholland ja"),
    ("Hij zei: \"nee\" — en ging.", "hij zei nee en ging"),
    ("", ""),
])
def test_normalize_dutch(raw, expected):
    assert normalize_text(raw, "nl") == expected


def test_reference_and_hypothesis_styles_agree():
    # FLEURS references are lowercased without punctuation; models emit
    # cased, punctuated text with digits. Both must normalize identically.
    ref = "op 15 augustus 1940 vielen de geallieerden zuid-frankrijk binnen"
    hyp = "Op 15 augustus 1940 vielen de geallieerden Zuid-Frankrijk binnen."
    assert normalize_text(ref, "nl") == normalize_text(hyp, "nl")


def _samples():
    return [
        {"reference": "de kat zit op de mat", "hypothesis": "De kat zit op de mat.", "rtf": 0.1},
        {"reference": "twee honden", "hypothesis": "2 katten", "rtf": 0.3},
        {"reference": "mislukt", "hypothesis": "", "error": "timeout"},
    ]


def test_score_samples():
    m = score_samples(_samples(), "nl")
    assert m["wer"] == 12.5  # 1 substitution / 8 reference words
    assert (m["n_samples"], m["n_total"], m["n_failed"]) == (2, 3, 1)
    assert m["rtf"] == 0.2


def test_rescore_updates_metrics_and_version():
    record = {"normalize": True, "results": {"fleurs_nl": {"wer": 99.0, "per_sample": _samples()}}}
    changes = rescore(record)
    assert changes == {"fleurs_nl": (99.0, 12.5)}
    assert record["normalizer_version"] == NORMALIZER_VERSION


def test_to_mono_layouts():
    left, right = np.ones(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32)
    expected = np.full(1000, 0.5, dtype=np.float32)
    assert np.array_equal(to_mono(left), left)                         # already mono
    assert np.array_equal(to_mono(np.stack([left, right])), expected)  # (channels, samples)
    assert np.array_equal(to_mono(np.stack([left, right], axis=1)), expected)  # (samples, channels)
    assert np.array_equal(to_mono(left[None, :]), left)                # (1, samples)
    with pytest.raises(ValueError):
        to_mono(np.zeros((2, 2, 10)))


def test_score_samples_exclude():
    from asr_nl.evaluation import score_samples

    samples = [
        {"index": 0, "reference": "een twee", "hypothesis": "een twee", "rtf": 0.1},
        {"index": 1, "reference": "drie vier", "hypothesis": "iets anders helemaal", "rtf": 0.1},
    ]
    assert score_samples([dict(s) for s in samples], "nl")["wer"] > 0
    metrics = score_samples([dict(s) for s in samples], "nl", exclude={1: "bad clip"})
    assert metrics["wer"] == 0 and metrics["excluded"] == [1] and metrics["n_samples"] == 1
