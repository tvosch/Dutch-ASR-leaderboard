"""Audio processing utilities."""

import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

TARGET_SR = 16_000


def audio_to_wav_bytes(array: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy float32 audio array to in-memory WAV bytes."""
    try:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, array.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    except ImportError:
        from scipy.io import wavfile
        pcm = (np.clip(array, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, sample_rate, pcm)
        return buf.getvalue()


def resample_to_16k(array: np.ndarray, sr: int) -> np.ndarray:
    """Resample audio to 16 kHz so all API backends receive a consistent sample rate."""
    if sr == TARGET_SR:
        return array.astype(np.float32)
    
    try:
        import torchaudio.functional as F
        import torch
        t = torch.from_numpy(array.astype(np.float32)).unsqueeze(0)
        t = F.resample(t, sr, TARGET_SR)
        return t.squeeze(0).numpy()
    except Exception:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, TARGET_SR)
        return resample_poly(array, TARGET_SR // g, sr // g).astype(np.float32)


def find_free_port() -> int:
    """Find an available TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
