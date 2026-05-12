"""Backend factory."""

import logging

from .api import AudioAPIBackend, VLLMServerBackend
from .base import BaseBackend
from .nemo import NeMoBackend
from .transformers import TransformersBackend

logger = logging.getLogger(__name__)

VLLM_AUDIO_ARCHITECTURES = {
    "Qwen2AudioForConditionalGeneration",
    "UltravoxModel",
    "WhisperForConditionalGeneration",
}


def create_backend(args) -> BaseBackend:
    """Create the backend specified by args.backend."""
    backend = args.backend

    if backend == "api":
        base_url = args.api_base_url or "https://api.openai.com/v1"
        logger.info(f"Using API backend: {base_url}")
        return AudioAPIBackend(args.model, base_url, args.api_key or "dummy", args.api_mode)

    if backend == "vllm":
        logger.info("Using vLLM backend")
        return VLLMServerBackend(
            args.model, args.device, args.dtype,
            args.tensor_parallel_size, args.vllm_port, args.api_mode,
            extra_args=args.vllm_args,
        )

    if backend == "transformers":
        logger.info("Using transformers backend")
        return TransformersBackend(args.model, args.device)

    if backend == "nemo":
        logger.info("Using NeMo backend")
        return NeMoBackend(args.model, args.device)

    raise ValueError(f"Unknown backend: {backend!r}")
