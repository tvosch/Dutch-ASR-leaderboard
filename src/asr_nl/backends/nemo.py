"""NVIDIA NeMo ASR backend."""

import logging
import tempfile
import time
from pathlib import Path

from asr_nl.audio import TARGET_SR, audio_to_wav_bytes, resample_to_16k

from .base import BaseBackend

logger = logging.getLogger(__name__)


class NeMoBackend(BaseBackend):
    """NeMo ASR backend using nemo.collections.asr."""

    def __init__(self, model_id: str, device: str = "cuda"):
        import nemo.collections.asr as nemo_asr

        logger.info(f"Loading NeMo model {model_id} on {device}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

        if device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.model.eval()
        logger.info("NeMo model loaded.")

    def transcribe(self, audio: dict, language: str = "nl") -> tuple[str, float]:
        array = audio["array"]
        sr = audio["sampling_rate"]
        duration = len(array) / sr

        array = resample_to_16k(array, sr)
        wav_bytes = audio_to_wav_bytes(array, TARGET_SR)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = Path(f.name)

        try:
            # Suppress NeMo's per-sample Lhotse/dataloader warnings
            noisy = [logging.getLogger(n) for n in ("nemo", "nemo_logger", "lhotse")]
            prev_levels = [lg.level for lg in noisy]
            for lg in noisy:
                lg.setLevel(logging.ERROR)
            t0 = time.perf_counter()
            try:
                results = self.model.transcribe([str(tmp_path)], batch_size=1, verbose=False)
            finally:
                elapsed = time.perf_counter() - t0
                for lg, lvl in zip(noisy, prev_levels):
                    lg.setLevel(lvl)
        finally:
            tmp_path.unlink(missing_ok=True)

        # NeMo returns a list; each element may be a string or a dataclass
        text = results[0]
        if not isinstance(text, str):
            text = text.text if hasattr(text, "text") else str(text)

        rtf = elapsed / duration if duration > 0 else 0.0
        return text, rtf

    def run_info(self) -> dict:
        import nemo
        return {"nemo_version": nemo.__version__}
