"""Editable Swahili/English prohibited-goods lexicon matcher.

Loads ``config/lexicon.json`` (see the ``_help`` block in that file) and matches it
against several normalised variants of the listing text, so disguised spellings
(``b4ng1``, ``b.a.n.g.i``, ``baaangi``) and typos (``bastolla``) are caught while
legitimate phrases on the allow lists (``unga wa sembe``, ``glue gun``) are not.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

from config.settings import CONFIG, DECISION_BLOCK, DECISION_REVIEW, LEXICON_PATH
from services.text_normalizer import basic_normalise, extract_emoji, normalise_variants, strip_punctuation

logger = logging.getLogger(__name__)

LEVEL_NONE = "NONE"
LEVEL_WEAK = "WEAK"

# Score attached to each lexicon level (used by score fusion and the meta-model)
LEVEL_SCORES: Dict[str, float] = {
    LEVEL_NONE: 0.0,
    LEVEL_WEAK: 0.25,
    DECISION_REVIEW: 0.60,
    DECISION_BLOCK: 0.95,
}

_SQUEEZE_RE = re.compile(r"([a-z])\1+")


def _squeeze(text: str) -> str:
    """Collapse repeated letters (``massage`` -> ``masage``)."""
    return _SQUEEZE_RE.sub(r"\1", text)


def _clean_term(term: str) -> str:
    """Normalise a lexicon entry the same way listing text is normalised."""
    return strip_punctuation(basic_normalise(term)) if any(c.isalnum() for c in term) else term


def _compile(terms: List[str], plural: bool = False) -> Optional[Pattern[str]]:
    """Compile a list of phrases into one word-bounded alternation regex.

    Parameters:
        terms (List[str]): Normalised phrases.
        plural (bool): Also accept a trailing ``s`` (used for allow phrases: "glue guns").

    Returns:
        Optional[Pattern[str]]: Compiled regex, or None for an empty list.
    """
    unique = sorted({t for t in terms if t}, key=len, reverse=True)
    if not unique:
        return None
    body = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in unique)
    suffix = "s?" if plural else ""
    return re.compile(rf"(?<![\w'])(?:{body}){suffix}(?![\w'])")


def _levenshtein(a: str, b: str) -> int:
    """Edit distance (uses rapidfuzz when installed, pure Python otherwise)."""
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    except ImportError:  # pragma: no cover - rapidfuzz is in requirements
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


@dataclass
class LexiconHit:
    """One matched lexicon entry."""

    category: str
    term: str
    tier: str  # strong | review | weak | context | weak_cancel | emoji | fuzzy
    method: str  # exact | deobfuscated | fuzzy | emoji


@dataclass
class CategoryResult:
    """Lexicon outcome for one category."""

    category: str
    level: str = LEVEL_NONE
    score: float = 0.0
    hits: List[LexiconHit] = field(default_factory=list)

    @property
    def terms(self) -> List[str]:
        """Matched terms (strong/review/weak/fuzzy) without context words."""
        return [h.term for h in self.hits if h.tier in ("strong", "review", "weak", "fuzzy")]


@dataclass
class LexiconResult:
    """Full lexicon analysis of one text."""

    categories: Dict[str, CategoryResult]
    youth_terms: List[str]
    emoji: List[str]
    obfuscation_detected: bool

    def worst(self) -> Optional[CategoryResult]:
        """Return the highest-scoring category, or None when nothing matched."""
        hits = [c for c in self.categories.values() if c.score > 0]
        return max(hits, key=lambda c: c.score) if hits else None

    def flagged(self, min_level: str = DECISION_REVIEW) -> List[CategoryResult]:
        """Categories at or above ``min_level`` (REVIEW or BLOCK)."""
        threshold = LEVEL_SCORES[min_level]
        return sorted(
            (c for c in self.categories.values() if c.score >= threshold), key=lambda c: -c.score
        )


@dataclass
class _CompiledCategory:
    """Pre-compiled regexes for one category (normal + squeezed variants)."""

    name: str
    label_en: str
    label_sw: str
    weak_with_context: str
    review_with_context: str
    patterns: Dict[str, Optional[Pattern[str]]]
    squeezed_patterns: Dict[str, Optional[Pattern[str]]]
    allow: Optional[Pattern[str]]
    emoji: List[str]
    fuzzy_terms: List[str]


class Lexicon:
    """Compiled lexicon with :meth:`analyse` for listing text."""

    TIERS: Tuple[str, ...] = ("strong", "review", "weak", "context", "weak_cancel")

    def __init__(self, data: Dict, fuzzy_min_length: int = 5) -> None:
        """Compile the lexicon.

        Parameters:
            data (Dict): Parsed lexicon.json content.
            fuzzy_min_length (int): Minimum term length for typo matching.
        """
        self.raw = data
        self.global_allow = _compile([_clean_term(t) for t in data.get("global_allow", [])], plural=True)
        self.global_allow_squeezed = _compile(
            [_squeeze(_clean_term(t)) for t in data.get("global_allow", [])], plural=True
        )
        self.fuzzy_exclude = {_clean_term(t) for t in data.get("fuzzy_exclude", [])}
        self.fuzzy_min_length = fuzzy_min_length
        self.categories: Dict[str, _CompiledCategory] = {}
        all_terms = set()

        for name, spec in data["categories"].items():
            tiers = {tier: [_clean_term(t) for t in spec.get(tier, [])] for tier in self.TIERS}
            all_terms.update(t for tier in ("strong", "review", "weak") for t in tiers[tier])
            self.categories[name] = _CompiledCategory(
                name=name,
                label_en=spec.get("label_en", name.lower()),
                label_sw=spec.get("label_sw", name.lower()),
                weak_with_context=spec.get("weak_with_context", DECISION_REVIEW),
                review_with_context=spec.get("review_with_context", DECISION_BLOCK),
                patterns={tier: _compile(terms) for tier, terms in tiers.items()},
                # Squeezed text only uses terms that stay distinctive once squeezed
                squeezed_patterns={
                    tier: _compile([_squeeze(t) for t in terms if len(_squeeze(t)) >= 4])
                    for tier, terms in tiers.items()
                },
                allow=_compile([_clean_term(t) for t in spec.get("allow", [])], plural=True),
                emoji=[e.replace("️", "") for e in spec.get("emoji", [])],
                fuzzy_terms=[
                    t for t in tiers["strong"] if " " not in t and len(t) >= fuzzy_min_length and t.isalpha()
                ],
            )
        self.known_terms = all_terms

        youth = data.get("youth_indicators", {})
        self.youth_pattern = _compile([_clean_term(t) for t in youth.get("terms", [])])
        self.youth_age_regex = re.compile(youth["age_regex"]) if youth.get("age_regex") else None
        self.seller_messages: Dict[str, str] = data.get("seller_messages", {})

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Path = LEXICON_PATH) -> "Lexicon":
        """Load and compile ``lexicon.json``.

        Parameters:
            path (Path): Path to the JSON lexicon.

        Returns:
            Lexicon: Compiled lexicon.
        """
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        lex = cls(data, fuzzy_min_length=int(CONFIG["text"].get("fuzzy_min_length", 5)))
        logger.info("Loaded lexicon from %s (%d categories)", path, len(lex.categories))
        return lex

    def label(self, category: str, language: str = "en") -> str:
        """Human-readable label of a category in English (``en``) or Swahili (``sw``)."""
        cat = self.categories.get(category)
        if cat is None:
            return category.lower()
        return cat.label_sw if language == "sw" else cat.label_en

    # ------------------------------------------------------------------
    def _fuzzy_hits(self, tokens: List[str], cat: _CompiledCategory) -> List[LexiconHit]:
        """Typo matching: same first letter and edit distance 1 (2 for long words)."""
        hits: List[LexiconHit] = []
        for token in set(tokens):
            if len(token) < self.fuzzy_min_length or token in self.fuzzy_exclude or token in self.known_terms:
                continue
            for term in cat.fuzzy_terms:
                if token[0] != term[0] or abs(len(token) - len(term)) > 2:
                    continue
                max_dist = 2 if len(term) >= 9 else 1
                if _levenshtein(token, term) <= max_dist:
                    hits.append(LexiconHit(cat.name, f"{token}~{term}", "fuzzy", "fuzzy"))
                    break
        return hits

    def analyse(self, text: str) -> LexiconResult:
        """Match the lexicon against ``text``.

        Parameters:
            text (str): Raw title + description (or OCR / transcript text).

        Returns:
            LexiconResult: Per-category levels and the matched evidence.
        """
        variants = normalise_variants(text)
        emoji_found = [e.replace("️", "") for e in extract_emoji(text)]
        results: Dict[str, CategoryResult] = {}
        obfuscated = False

        for cat in self.categories.values():
            found: Dict[Tuple[str, str], LexiconHit] = {}
            for kind, variant in variants:
                squeezed = kind == "squeezed"
                allow_re = self.global_allow_squeezed if squeezed else self.global_allow
                cleaned = allow_re.sub(" . ", variant) if allow_re else variant
                if cat.allow:
                    cleaned = cat.allow.sub(" . ", cleaned)
                patterns = cat.squeezed_patterns if squeezed else cat.patterns
                for tier, pattern in patterns.items():
                    if pattern is None:
                        continue
                    for match in pattern.finditer(cleaned):
                        term = re.sub(r"\s+", " ", match.group(0))
                        key = (tier, _squeeze(term))
                        if key not in found:
                            method = "exact" if kind == "plain" else "deobfuscated"
                            found[key] = LexiconHit(cat.name, term, tier, method)
                if kind in ("deobfuscated", "plain") and cat.fuzzy_terms:
                    for hit in self._fuzzy_hits(cleaned.split(), cat):
                        found.setdefault(("fuzzy", hit.term), hit)

            for emo in emoji_found:
                if emo in cat.emoji:
                    found.setdefault(("emoji", emo), LexiconHit(cat.name, emo, "emoji", "emoji"))

            result = self._score_category(cat, list(found.values()))
            if result.score > 0 or result.hits:
                results[cat.name] = result
                obfuscated = obfuscated or any(
                    h.method in ("deobfuscated", "fuzzy") and h.tier in ("strong", "review", "fuzzy")
                    for h in result.hits
                )

        youth_terms: List[str] = []
        plain = variants[0][1] if variants else ""
        if self.youth_pattern is not None:
            youth_terms.extend(m.group(0) for m in self.youth_pattern.finditer(plain))
        if self.youth_age_regex is not None:
            youth_terms.extend(m.group(0) for m in self.youth_age_regex.finditer(plain))

        return LexiconResult(
            categories=results,
            youth_terms=sorted(set(youth_terms)),
            emoji=emoji_found,
            obfuscation_detected=obfuscated,
        )

    @staticmethod
    def _score_category(cat: _CompiledCategory, hits: List[LexiconHit]) -> CategoryResult:
        """Turn the hits of one category into a level and score."""
        result = CategoryResult(category=cat.name, hits=hits)
        tiers = {h.tier for h in hits}
        weak_terms = {_squeeze(h.term) for h in hits if h.tier == "weak"}
        matched_terms = [f" {_squeeze(h.term)} " for h in hits if h.tier in ("weak", "review", "strong")]
        # a context word only counts when it is not part of the matched term itself ("frp bypass" / "bypass")
        context_terms = {
            t for t in (_squeeze(h.term) for h in hits if h.tier == "context")
            if not any(f" {t} " in m for m in matched_terms)
        }
        has_context = bool(context_terms) or "emoji" in tiers
        if not weak_terms and "emoji" in tiers and context_terms:
            # emoji used as the noun ("nina 🍁 ya kuvuta"): emoji + a text context word = weak + context
            weak_terms = {h.term for h in hits if h.tier == "emoji"}

        if weak_terms and not has_context and "weak_cancel" in tiers:
            # innocent context (e.g. "unga safi wa sembe"): drop the lone weak words
            hits = [h for h in hits if h.tier not in ("weak", "weak_cancel")]
            result.hits = hits
            tiers = {h.tier for h in hits}
            weak_terms = set()

        if "strong" in tiers:
            result.level = DECISION_BLOCK
        elif "review" in tiers and has_context:
            result.level = cat.review_with_context  # e.g. "silaha" + "bila leseni" -> BLOCK for WEAPONS
        elif "review" in tiers or "fuzzy" in tiers:
            result.level = DECISION_REVIEW
        elif weak_terms and has_context:
            result.level = cat.weak_with_context
        elif weak_terms:
            result.level = LEVEL_WEAK
        else:
            result.level = LEVEL_NONE

        base = LEVEL_SCORES[result.level]
        if result.level == DECISION_BLOCK and "strong" not in tiers:
            base = 0.85  # weak word + context: confident but below an explicit strong term
        extra = sum(1 for h in hits if h.tier in ("strong", "review", "fuzzy")) - 1
        result.score = round(min(1.0, base + 0.02 * max(0, extra)), 4) if base else 0.0
        return result


_lexicon_lock = threading.Lock()
_lexicon: Optional[Lexicon] = None


def get_lexicon(reload: bool = False) -> Lexicon:
    """Return the process-wide lexicon, loading (or reloading) it on demand.

    Parameters:
        reload (bool): Re-read lexicon.json from disk.

    Returns:
        Lexicon: Compiled lexicon.
    """
    global _lexicon
    with _lexicon_lock:
        if _lexicon is None or reload:
            _lexicon = Lexicon.from_file()
        return _lexicon
