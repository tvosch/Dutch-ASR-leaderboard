"""Base class for ASR backends."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseBackend(ABC):
    """Abstract base class for ASR backends."""
    
    @abstractmethod
    def transcribe(self, audio: dict, language: str = "nl") -> tuple[str, float]:
        """
        Transcribe audio and return (transcript, rtf).
        
        Args:
            audio: Dict with 'array' (np.ndarray) and 'sampling_rate' (int)
            language: Language code (default: "nl")
            
        Returns:
            Tuple of (transcript text, real-time factor)
        """
        pass
    
    def close(self):
        """Cleanup resources."""
        logger.debug("Closing backend")
        pass
