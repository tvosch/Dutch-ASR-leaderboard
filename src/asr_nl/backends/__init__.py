"""Backends package for ASR evaluation."""

from .api import AudioAPIBackend, VLLMServerBackend
from .base import BaseBackend
from .factory import create_backend, VLLM_AUDIO_ARCHITECTURES
from .nemo import NeMoBackend
from .transformers import TransformersBackend

__all__ = [
    "BaseBackend",
    "AudioAPIBackend",
    "VLLMServerBackend",
    "TransformersBackend",
    "NeMoBackend",
    "create_backend",
    "VLLM_AUDIO_ARCHITECTURES",
]
