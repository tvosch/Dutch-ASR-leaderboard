"""Transformers pipeline backend."""

import logging
import time

logger = logging.getLogger(__name__)


class TransformersBackend:
    """HuggingFace transformers pipeline backend."""
    
    def __init__(self, model_id: str, device: str = "cpu"):
        from transformers import pipeline

        device_arg = 0 if device == "cuda" else -1
        import torch
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(f"Loading transformers pipeline for {model_id} on {device}")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device_arg,
            torch_dtype=torch_dtype,
            chunk_length_s=30,
        )
    
    def transcribe(self, audio: dict, language: str = "nl") -> tuple[str, float]:
        """Transcribe using transformers pipeline."""
        array = audio["array"]
        sr = audio["sampling_rate"]
        duration = len(array) / sr
        
        t0 = time.perf_counter()
        
        try:
            out = self.pipe(
                {"array": array, "sampling_rate": sr},
                generate_kwargs={"language": language, "task": "transcribe"},
            )
        except TypeError:
            # CTC models don't accept generate_kwargs
            out = self.pipe({"array": array, "sampling_rate": sr})
        
        rtf = (time.perf_counter() - t0) / duration if duration > 0 else 0.0
        return out["text"], rtf
