"""Text normalization utilities.

Bump NORMALIZER_VERSION whenever the output of normalize_text() changes, so
results scored with different normalizers are never mixed. Use `asr-nl-rescore`
to re-score existing result files after a bump.

Version 1 spelled digits out as words. Version 2 does the opposite: spelled-out
numbers become digits, so "twintig procent", "20 procent" and "20%" all score
the same. Version 1 is kept so older scores can be reproduced.
"""

import logging
import re
import string
import unicodedata

from text_to_num import alpha2digit

from .dutch_rules import CONTRACTIONS, UNITS, YEAR_CONTEXT

logger = logging.getLogger(__name__)

NORMALIZER_VERSION = 2

FILLER_WORDS = {"uh", "uhm", "um", "eh", "ehm", "euh", "hm", "hmm", "mhm", "mm", "ah"}


# ---------------------------------------------------------------------------
# Version 1 (frozen: do not change, it reproduces the original leaderboard)
# ---------------------------------------------------------------------------

_FILLER_WORDS_V1 = {"uh", "um", "eh", "euh", "hmm", "hm", "ah", "uhm"}

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


def normalize_dutch_v1(text: str) -> str:
    """Normalize Dutch text for evaluation (version 1)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Strip ASCII and Unicode punctuation
    text = "".join(
        c for c in text
        if not unicodedata.category(c).startswith("P") and c not in string.punctuation
    )
    text = _expand_numbers(text)
    words = [w for w in text.split() if w not in _FILLER_WORDS_V1]
    return " ".join(words).strip()


# ---------------------------------------------------------------------------
# Version 2
# ---------------------------------------------------------------------------

_APOSTROPHES = "'’‘`´ʼ"
_SYMBOL_WORDS_NL = {"%": " procent ", "&": " en ", "+": " plus ", "°": " graden "}
_CURRENCIES = {"€": "euro", "$": "dollar", "£": "pond"}  # spelled out only next to an amount
_SCALES = ["miljoen", "miljard", "biljoen"]
_SCALE_RE = re.compile(r"\b(miljoen|miljard|biljoen)(?:en)?\b")                     # kept as words
_CURRENCY_FIRST_RE = re.compile(r"([€$£])\s?(\d[\d.,]*\d|\d)(\s+(?:miljoen|miljard|biljoen))?")  # €3,5 miljoen -> 3,5 miljoen euro
_CURRENCY_AFTER_RE = re.compile(r"(\d)\s?([€$£])")                                          # 5€ -> 5 euro
_UNIT_EN_TENS_RE = re.compile(
    r"\b(een|één|twee|drie|vier|vijf|zes|zeven|acht|negen)[\s-]+en[\s-]+"
    r"(twintig|dertig|veertig|vijftig|zestig|zeventig|tachtig|negentig)(?=ste\b|\b)"
)                                                                                     # acht en twintig -> achtentwintig
_LETTER_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-(?=[^\W\d_]|\s)")                     # veertig-duizend, drie- en vier
_FRACTION_RE = re.compile(r"(\d)([¼-¾⅐-⅞])")                                         # 29¾ -> 29 ¾, before NFKC
_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b(?![.,]\d)")                     # 10.000 -> 10000
_TIME_RE = re.compile(r"\b([01]?\d|2[0-4])[:.]([0-5]\d)(?:\s*(?:uur|u)\b|\b)")        # 10:31, 10.31 uur, 5.30u
_DECIMAL_RE = re.compile(r"(\d),(\d)")                                               # 3,5 -> 3 komma 5
_POINT_RE = re.compile(r"(\d)\.(\d)")                                                # 3.5 -> 3 punt 5
_ORDINAL_RE = re.compile(r"\b(\d+)(?:ste|de|e)\b")                                    # 21ste / 21de -> 21e
_DIGIT_DASH_RE = re.compile(r"(\d)\s*[-–—]\s*(\d)")                                   # 1-3 -> 1 3
_DIGIT_LETTER_RE = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")             # m16 -> m 16
_UNIT_RE = re.compile(
    r"(\d)\s*(" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True)) + r")(?!\w)"
)
_SPOKEN_YEAR_RE = re.compile(r"\b(\w+) (1[1-9]|20) ([1-9]\d)\b")                      # in 19 87 -> in 1987


def _compile_rules(rules: dict[str, str]):
    """One alternation over all patterns (longest first), applied in a single
    pass so a replacement cannot trigger another rule."""
    items = sorted(rules.items(), key=lambda kv: -len(kv[0]))
    pattern = re.compile("|".join(f"(?P<_{i}>{pat})" for i, (pat, _) in enumerate(items)))
    replacements = [rep for _, rep in items]
    return lambda text: pattern.sub(lambda m: replacements[int(m.lastgroup[1:])], text)


_expand_contractions = _compile_rules(CONTRACTIONS)


def _fold_diacritics(text: str, keep: str = "") -> str:
    """é -> e, ë -> e, ... (stress accents are optional in Dutch spelling).
    Combining marks in `keep` are left in place."""
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c) or c in keep)
    return unicodedata.normalize("NFC", folded)


def _time(m: re.Match) -> str:
    hour, minute = int(m.group(1)), m.group(2)
    return f" {hour} uur " if minute == "00" else f" {hour} uur {int(minute)} "  # 10:31 -> 10 uur 31


def _spoken_year(m: re.Match) -> str:
    word, century, rest = m.groups()
    if int(rest) >= 60 or word in YEAR_CONTEXT:
        return f"{word} {century}{rest}"
    return m.group()  # "om 19 45" is a clock time


def normalize_dutch(text: str) -> str:
    """Normalize Dutch text for evaluation (version 2).

    Lowercases, expands contractions, writes numbers as digits ("twintig" and
    "20" both become "20"), spells out symbols, currencies and units next to a
    number ("20%" -> "20 procent", "35mm" -> "35 millimeter"), folds
    diacritics, removes punctuation and filler words.
    """
    text = _FRACTION_RE.sub(r"\1 \2", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join("'" if c in _APOSTROPHES else c for c in text)
    text = _expand_contractions(text)  # z'n -> zijn, 't -> het, ...
    text = _CURRENCY_FIRST_RE.sub(lambda m: f"{m.group(2)}{m.group(3) or ''} {_CURRENCIES[m.group(1)]}", text)
    text = _CURRENCY_AFTER_RE.sub(lambda m: f"{m.group(1)} {_CURRENCIES[m.group(2)]}", text)

    # Spelled-out numbers -> digits. Scale words stay words ("3 miljoen" == "drie miljoen");
    # "en" between other words is protected so "vijf en honderd" stays two numbers.
    # Stress accents go first ("honderdéen"), but not the diaeresis: text2num needs
    # it in "tweeëntwintigste".
    text = _fold_diacritics(text, keep="̈")
    text = _UNIT_EN_TENS_RE.sub(r"\1en\2", text)
    text = _LETTER_HYPHEN_RE.sub("", text)
    text = _SCALE_RE.sub(lambda m: f" zzscale{_SCALES.index(m.group(1))}zz ", text)
    text = re.sub(r"\ben\b", " zzenzz ", text)
    text = alpha2digit(text, "nl", threshold=0)  # threshold 0: also 0-2 when on their own
    text = text.replace("zzenzz", "en")
    text = re.sub(r"zzscale(\d)zz", lambda m: _SCALES[int(m.group(1))], text)

    # Digit formats, while punctuation is still there.
    text = _THOUSANDS_RE.sub(lambda m: m.group().replace(".", ""), text)
    text = _TIME_RE.sub(_time, text)
    text = _DECIMAL_RE.sub(r"\1 komma \2", text)
    text = _POINT_RE.sub(r"\1 punt \2", text)
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {UNITS[m.group(2)]} ", text)
    for sym, word in _SYMBOL_WORDS_NL.items():
        text = text.replace(sym, word)
    text = _ORDINAL_RE.sub(r"\1e", text)
    text = _DIGIT_DASH_RE.sub(r"\1 \2", text)

    # Remaining apostrophes and intra-word hyphens join ("auto's" -> "autos",
    # "noord-holland" -> "noordholland"); other punctuation and symbols separate words.
    text = _fold_diacritics(text)
    text = "".join("" if c in "'-" else c for c in text)
    text = _DIGIT_LETTER_RE.sub(" ", text)
    text = re.sub(r"\b(\d+) e\b", r"\1e", text)  # keep ordinals one word: 21e
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    text = " ".join(text.split())
    text = _SPOKEN_YEAR_RE.sub(_spoken_year, text)

    words = []
    for w in text.split():
        if w in FILLER_WORDS:
            continue
        words.append("een" if w == "1" else w)  # "een" is both the article and the number
    return " ".join(words)


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


_DUTCH_NORMALIZERS = {1: normalize_dutch_v1, 2: normalize_dutch}


def normalize_text(text: str, language: str = "nl", version: int = NORMALIZER_VERSION) -> str:
    """Normalize text based on language. `version` selects the Dutch normalizer."""
    if not text:
        return ""
    if language == "en":
        return _get_english_normalizer()(text)
    return _DUTCH_NORMALIZERS[version](text)
