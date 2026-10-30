"""murmel-flash backend (open-weights Dutch ASR, requires the murmel-flash package)."""

import logging
import time

from asr_nl.audio import TARGET_SR, resample_to_16k

from .base import BaseBackend

logger = logging.getLogger(__name__)


class MurmelFlashBackend(BaseBackend):
    """murmel-flash through its own package, with the package defaults (attention beam search, beam 4)."""

    def __init__(self, model_id: str, device: str = "cuda"):
        from murmel_flash.inference import Transcriber

        logger.info(f"Loading murmel-flash model {model_id} on {device}")
        self.device = device
        self.model = Transcriber.from_pretrained(model_id, device=device)
        logger.info("murmel-flash model loaded.")

    def _sync(self):
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()

    def transcribe(self, audio: dict, language: str = "nl") -> tuple[str, float]:
        array = resample_to_16k(audio["array"], audio["sampling_rate"])
        duration = len(array) / TARGET_SR

        self._sync()
        t0 = time.perf_counter()
        out = self.model.transcribe(array)
        self._sync()
        elapsed = time.perf_counter() - t0

        rtf = elapsed / duration if duration > 0 else 0.0
        return out["text"].strip(), rtf

    def run_info(self) -> dict:
        from importlib.metadata import version
        return {"murmel_flash_version": version("murmel-flash"), "decoding": "beam 4 (package default)"}
