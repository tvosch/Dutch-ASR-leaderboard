"""Base class for ASR backends."""

from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def transcribe(self, audio: dict, language: str = "nl") -> tuple[str, float]:
        """
        Transcribe audio and return (transcript, rtf).

        Args:
            audio: Dict with 'array' (np.ndarray) and 'sampling_rate' (int)
            language: Language code (default: "nl")
        """

    def run_info(self) -> dict:
        """Versions and settings recorded in the result file for reproducibility."""
        return {}

    def close(self):
        """Release resources (servers, GPU memory)."""
