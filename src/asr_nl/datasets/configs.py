"""Dataset configurations for ASR evaluation."""

from typing import Any

DATASETS: dict[str, dict[str, Any]] = {
    "fleurs": {
        "hf_id": "google/fleurs",
        "config": "nl_nl",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcription",
        "key": "fleurs_nl",
        "trust_remote_code": True,
    },
    "fleurs_en": {
        "hf_id": "google/fleurs",
        "config": "en_us",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcription",
        "key": "fleurs_en",
        "trust_remote_code": True,
    },
    "common_voice": {
        "hf_id": "mozilla-foundation/common_voice_18_0",
        "config": "nl",
        "split": "test",
        "audio_col": "audio",
        "text_col": "sentence",
        "key": "common_voice_18_nl",
        "trust_remote_code": True,
    },
    "common_voice_25": {
        "hf_id": "mozilla-foundation/common_voice_25_0",
        "config": "nl",
        "split": "test",
        "audio_col": "audio",
        "text_col": "sentence",
        "key": "common_voice_25_nl",
        "trust_remote_code": True,
        "license": "CC0-1.0",
        "notes": (
            "Gated on HF Hub (click-through terms). Also distributed ungated as a tar "
            "by the Mozilla Data Collective; use --save-to-disk Arrow snapshots for offline eval. "
            "Same-version train/test splits are speaker-disjoint."
        ),
    },
    "voxpopuli": {
        "hf_id": "facebook/voxpopuli",
        "config": "nl",
        "split": "test",
        "audio_col": "audio",
        "text_col": "normalized_text",
        "key": "voxpopuli_nl",
    },
    "voxpopuli_en": {
        "hf_id": "facebook/voxpopuli",
        "config": "en",
        "split": "test",
        "audio_col": "audio",
        "text_col": "normalized_text",
        "key": "voxpopuli_en",
    },
    "mls_nl": {
        "hf_id": "facebook/multilingual_librispeech",
        "config": "dutch",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcript",
        "key": "mls_nl",
    },
}
