"""Dataset configurations for ASR evaluation.

Each dataset is pinned to a Hugging Face commit (`revision`) so an upstream
update cannot silently change the test set.
"""

from typing import Any

DATASETS: dict[str, dict[str, Any]] = {
    "fleurs": {
        "hf_id": "google/fleurs",
        "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "config": "nl_nl",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcription",
        "key": "fleurs_nl",
        "language": "nl",
        "trust_remote_code": True,
    },
    "fleurs_en": {
        "hf_id": "google/fleurs",
        "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd",
        "config": "en_us",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcription",
        "key": "fleurs_en",
        "language": "en",
        "trust_remote_code": True,
    },
    "common_voice": {
        "hf_id": "mozilla-foundation/common_voice_18_0",
        "config": "nl",
        "split": "test",
        "audio_col": "audio",
        "text_col": "sentence",
        "key": "common_voice_18_nl",
        "language": "nl",
        "trust_remote_code": True,
    },
    "voxpopuli": {
        "hf_id": "facebook/voxpopuli",
        "revision": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
        "config": "nl",
        "split": "test",
        "audio_col": "audio",
        "text_col": "normalized_text",
        "key": "voxpopuli_nl",
        "language": "nl",
    },
    "voxpopuli_en": {
        "hf_id": "facebook/voxpopuli",
        "revision": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
        "config": "en",
        "split": "test",
        "audio_col": "audio",
        "text_col": "normalized_text",
        "key": "voxpopuli_en",
        "language": "en",
    },
    "mls_nl": {
        "hf_id": "facebook/multilingual_librispeech",
        "revision": "2e83e61823b4c47dcbcb1980bb88601274127609",
        "config": "dutch",
        "split": "test",
        "audio_col": "audio",
        "text_col": "transcript",
        "key": "mls_nl",
        "language": "nl",
    },
}
