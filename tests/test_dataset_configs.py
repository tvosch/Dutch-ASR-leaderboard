"""Tests for dataset configuration entries."""

import pytest

from asr_nl.datasets import DATASETS

REQUIRED_KEYS = {"hf_id", "config", "split", "audio_col", "text_col", "key"}


@pytest.mark.parametrize("name", sorted(DATASETS.keys()))
def test_config_has_required_keys(name):
    """Every dataset entry must carry the keys the eval loop reads."""
    missing = REQUIRED_KEYS - set(DATASETS[name].keys())
    assert not missing, f"{name} is missing keys: {sorted(missing)}"


def test_result_keys_are_unique():
    """Result JSON keys must not collide across datasets."""
    keys = [cfg["key"] for cfg in DATASETS.values()]
    assert len(keys) == len(set(keys)), f"Duplicate result keys: {keys}"


def test_common_voice_25_entry():
    """CV 25 NL entry matches the Common Voice schema."""
    cfg = DATASETS["common_voice_25"]
    assert cfg["hf_id"] == "mozilla-foundation/common_voice_25_0"
    assert cfg["config"] == "nl"
    assert cfg["split"] == "test"
    assert cfg["text_col"] == "sentence"
    assert cfg["key"] == "common_voice_25_nl"
