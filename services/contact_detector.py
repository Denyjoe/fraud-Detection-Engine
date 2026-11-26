"""Off-platform contact detection.

Flags listings that push buyers off the store (where escrow and buyer protection do
not apply): Tanzanian phone numbers (incl. spaced/dotted or written out in Swahili
words), WhatsApp / Telegram invitations, external links and mobile-money "Lipa Namba"
requests. A bare product feature such as "WhatsApp inafanya kazi" is not flagged.
"""

import re
from dataclasses import dataclass, field
from typing import List

from services.text_normalizer import basic_normalise

# +255 7XX XXX XXX / 0 7XX XXX XXX / 06X..., with spaces, dots, commas or dashes (speech transcripts use commas)
_TZ_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\s?255|0)[\s.,\-]*[67](?:[\s.,\-]*\d){8}(?!\d)")
_INTL_PHONE_RE = re.compile(r"(?<![\w+])\+\s?(?!255)\d{1,3}(?:[\s.\-]*\d){7,12}(?!\d)")
_NUMBER_WORDS = {
    "sifuri", "zero", "moja", "one", "mbili", "two", "tatu", "three", "nne", "four", "tano", "five",
    "sita", "six", "saba", "seven", "nane", "eight", "tisa", "nine",
}
_APPS_RE = re.compile(
    r"\b(?:whats\s?app|whatsap+|watsap+|wats\s?app|wasap+|whatsup|w/?app|wa\.me|wtsp|telegram|telegrm|"
    r"t\.me|signal app|snapchat|imo app|messenger|instagram|insta|tiktok)\b"
)
_INVITE_RE = re.compile(
    r"\b(?:ni\s?cheki|nicheki|cheki|nitafute|nitumie|ni\s?dm|dm|inbox|chat|contact|message|meseji|ujumbe|text|"
    r"piga|nipigie|call|namba|number|no\.?|link|join|jiunge|group|kundi|channel|nifollow|follow|tuwasiliane|"
    r"wasiliana)\b"
)
_HANDLE_RE = re.compile(r"(?<![\w.])@[a-z0-9_.]{3,}")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[a-z0-9\-]+\.(?:com|net|org|co\.tz|tz|me|ly|ee|io|app|link)\b|\bbit\.ly/\S+"
)
_PAYMENT_RE = re.compile(
    r"\b(?:lipa\s+namba|lipa\s+kwa\s+(?:m-?pesa|tigo\s?pesa|airtel\s?money|halo\s?pesa|mixx)|"
    r"tuma\s+(?:pesa|hela)\s+kwa|tuma\s+kwa\s+namba|pay\s+(?:via|to)\s+(?:m-?pesa|airtel|tigo)|"
    r"m-?pesa\s+(?:namba|number|no))\b"
)


@dataclass
class ContactResult:
    """Off-platform contact evidence found in a text."""

    phones: List[str] = field(default_factory=list)
    apps: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    handles: List[str] = field(default_factory=list)
    payment_requests: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """True when any off-platform contact evidence was found."""
        return bool(self.phones or self.apps or self.urls or self.handles or self.payment_requests)

    @property
    def score(self) -> float:
        """Risk score 0..1 (payment requests and phone numbers weigh most)."""
        score = 0.0
        if self.payment_requests:
            score = max(score, 0.75)
        if self.phones:
            score = max(score, 0.65)
        if self.apps or self.handles:
            score = max(score, 0.55)
        if self.urls:
            score = max(score, 0.45)
        return score

    def summary(self) -> str:
        """Short English description of the evidence."""
        parts = []
        if self.phones:
            parts.append(f"phone number(s) {', '.join(self.phones[:3])}")
        if self.apps:
            parts.append(f"messaging app invitation ({', '.join(sorted(set(self.apps)))})")
        if self.handles:
            parts.append(f"social handle(s) {', '.join(self.handles[:3])}")
        if self.urls:
            parts.append(f"external link(s) {', '.join(self.urls[:3])}")
        if self.payment_requests:
            parts.append(f"off-platform payment request ('{self.payment_requests[0]}')")
        return "; ".join(parts)


def _spelled_out_numbers(text: str) -> List[str]:
    """Find phone numbers written as 7+ consecutive number words (``sifuri saba moja ...``)."""
    found: List[str] = []
    run: List[str] = []
    for token in re.findall(r"[a-z]+", text) + [""]:
        if token in _NUMBER_WORDS:
            run.append(token)
            continue
        if len(run) >= 7:
            found.append(" ".join(run))
        run = []
    return found


def detect_contacts(text: str) -> ContactResult:
    """Detect off-platform contact attempts in ``text``.

    Parameters:
        text (str): Listing text (title, description, OCR text or transcript).

    Returns:
        ContactResult: Evidence found (empty when the text is clean).
    """
    norm = basic_normalise(text)
    result = ContactResult()
    result.phones = [re.sub(r"[\s.,\-]", "", m.group(0)) for m in _TZ_PHONE_RE.finditer(norm)]
    result.phones += [re.sub(r"[\s.\-]", "", m.group(0)) for m in _INTL_PHONE_RE.finditer(norm)]
    result.phones += _spelled_out_numbers(norm)
    result.urls = [m.group(0) for m in _URL_RE.finditer(norm)]
    result.handles = [m.group(0) for m in _HANDLE_RE.finditer(norm)]
    result.payment_requests = [m.group(0) for m in _PAYMENT_RE.finditer(norm)]

    # A messaging app only counts when it is an invitation (nearby invite word, phone, handle or link)
    for sentence in re.split(r"[.!?\n;]", norm):
        for match in _APPS_RE.finditer(sentence):
            app = match.group(0)
            if app in ("wa.me", "t.me") or _INVITE_RE.search(sentence) or _HANDLE_RE.search(sentence) \
                    or _TZ_PHONE_RE.search(sentence) or result.phones:
                result.apps.append(app)
    return result
