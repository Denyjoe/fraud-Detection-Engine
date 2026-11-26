"""Text normalisation and de-obfuscation for Swahili/English listings.

Sellers hide prohibited words with tricks such as ``b4ng1``, ``b.a.n.g.i``,
``B A N G I``, ``baaangi``, homoglyphs (Cyrillic ``а``), zero-width characters,
or emoji. :func:`normalise_variants` produces a small set of normalised versions of
the text so the lexicon can match all of them while still matching the plain text
exactly (so prices and model numbers are not mangled in the plain variant).
"""

import re
import unicodedata
from typing import List, Tuple

# Zero-width and formatting characters used to split words invisibly
_INVISIBLE_RE = re.compile("[​‌‍⁠﻿­]")

# Common homoglyphs (Cyrillic / Greek letters that look Latin)
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "ѕ": "s", "ԁ": "d", "ɡ": "g",
    "α": "a", "ο": "o", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "τ": "t", "ρ": "p",
})

# Leetspeak digits/symbols -> letters (applied only inside tokens that contain letters)
_LEET = str.maketrans({
    "4": "a", "@": "a", "3": "e", "1": "i", "!": "i", "|": "i", "0": "o", "5": "s", "$": "s",
    "7": "t", "+": "t", "8": "b", "9": "g",
})

# Single letters separated by dots/spaces/dashes etc.: "b.a.n.g.i", "b a n g i", "b-a-n-g-i"
_SPACED_LETTERS_RE = re.compile(r"(?<![\w])((?:[a-z][\s.\-_*•·|/\\,]{1,2}){2,}[a-z])(?![\w])")
_SEPARATORS_RE = re.compile(r"[\s.\-_*•·|/\\,]")

# Three or more identical letters: "baaangi" -> "bangi" (keeps Swahili double vowels like "saa")
_REPEAT3_RE = re.compile(r"([a-z])\1{2,}")
_REPEAT2_RE = re.compile(r"([a-z])\1+")

_TOKEN_WITH_DIGITS_RE = re.compile(r"[a-z0-9@$!|+]*[a-z][a-z0-9@$!|+]*")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s'+#%]", re.UNICODE)


def basic_normalise(text: str) -> str:
    """Lower-case, strip accents/invisible characters and map homoglyphs.

    Parameters:
        text (str): Raw text.

    Returns:
        str: Normalised text with punctuation preserved.
    """
    if not text:
        return ""
    text = _INVISIBLE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text).lower().translate(_HOMOGLYPHS)
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    text = text.replace("’", "'").replace("`", "'")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _join_spaced_letters(text: str) -> str:
    """Join runs of single letters separated by punctuation or spaces.

    Parameters:
        text (str): Normalised text.

    Returns:
        str: Text where ``b.a.n.g.i`` became ``bangi``.
    """
    return _SPACED_LETTERS_RE.sub(lambda m: _SEPARATORS_RE.sub("", m.group(1)), text)


def _decode_leet(text: str) -> str:
    """Replace leetspeak characters inside tokens that mix letters with digits/symbols.

    Pure numbers (prices, phone numbers) are left untouched.

    Parameters:
        text (str): Normalised text.

    Returns:
        str: Text with ``b4ng1`` decoded to ``bangi``.
    """

    def repl(match: re.Match) -> str:
        token = match.group(0)
        if token.isalpha():
            return token
        return token.translate(_LEET)

    return _TOKEN_WITH_DIGITS_RE.sub(repl, text)


def strip_punctuation(text: str) -> str:
    """Replace punctuation with spaces and collapse whitespace.

    Parameters:
        text (str): Text.

    Returns:
        str: Text containing only word characters and spaces.
    """
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub(" ", text)).strip()


def normalise_variants(text: str) -> List[Tuple[str, str]]:
    """Return de-duplicated, named normalised variants of ``text`` for lexicon matching.

    Variants (in order):
        1. plain: lower-case, accents/homoglyphs removed, punctuation -> spaces
        2. joined: spaced/dotted single letters joined
        3. deobfuscated: joined + leetspeak decoded + triple letters squeezed
        4. squeezed: every repeated letter collapsed (``baangi`` -> ``bangi``)

    Parameters:
        text (str): Raw listing text.

    Returns:
        List[Tuple[str, str]]: ``(kind, text)`` pairs; the first one is always ``plain``.
    """
    base = basic_normalise(text)
    plain = strip_punctuation(base)
    joined = strip_punctuation(_join_spaced_letters(base))
    deobfuscated = _REPEAT3_RE.sub(r"\1", strip_punctuation(_decode_leet(_join_spaced_letters(base))))
    squeezed = _REPEAT2_RE.sub(r"\1", deobfuscated)

    variants: List[Tuple[str, str]] = []
    seen = set()
    for kind, candidate in (("plain", plain), ("joined", joined), ("deobfuscated", deobfuscated),
                            ("squeezed", squeezed)):
        if candidate and candidate not in seen:
            seen.add(candidate)
            variants.append((kind, candidate))
    return variants


def extract_emoji(text: str) -> List[str]:
    """Return the emoji / pictographic characters found in ``text``.

    Parameters:
        text (str): Raw text.

    Returns:
        List[str]: Emoji characters in order of appearance (variation selectors removed).
    """
    found: List[str] = []
    for ch in text or "":
        if unicodedata.category(ch) == "So" or 0x1F000 <= ord(ch) <= 0x1FAFF:
            found.append(ch)
    return found
