"""Text normalization utilities."""

import logging
import re
import string
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

FILLER_WORDS = {"uh", "um", "eh", "euh", "hmm", "hm", "ah", "uhm"}

_NUM2WORDS_AVAILABLE = False
try:
    import num2words as _n2w
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    pass


def _expand_numbers(text: str) -> str:
    """Expand digits to number words (Dutch)."""
    if not _NUM2WORDS_AVAILABLE:
        logger.debug("num2words not available, skipping number expansion")
        return text

    def _replace(match: re.Match) -> str:
        try:
            return _n2w.num2words(int(match.group()), lang="nl")
        except Exception as e:
            logger.warning(f"Failed to expand number {match.group()}: {e}")
            return match.group()

    return re.sub(r"\b\d+\b", _replace, text)


def normalize_dutch(text: str) -> str:
    """Normalize Dutch text for evaluation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Strip ASCII and Unicode punctuation
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("P") and c not in string.punctuation
    )
    text = _expand_numbers(text)
    words = [w for w in text.split() if w not in FILLER_WORDS]
    return " ".join(words).strip()


# Lazy load Whisper English normalizer
_whisper_english_normalizer = None


def _get_english_normalizer():
    global _whisper_english_normalizer
    if _whisper_english_normalizer is None:
        try:
            from whisper.normalizers import EnglishTextNormalizer
            _whisper_english_normalizer = EnglishTextNormalizer()
            logger.debug("Loaded Whisper English normalizer")
        except ImportError:
            logger.warning("Whisper not available, using basic English normalization")
            _whisper_english_normalizer = lambda x: x.lower()
    return _whisper_english_normalizer


def normalize_text(text: str, language: str = "nl") -> str:
    """Normalize text based on language."""
    if language == "en":
        return _get_english_normalizer()(text)
    return normalize_dutch(text)
