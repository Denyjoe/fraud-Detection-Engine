"""Child-safety controls (mandatory).

1. **Escalation rule**: any sexual signal (text or image) combined with any youth
   indicator (ordinary age words in the text, or the age estimator on an image that
   already has a sexual signal) returns ``ESCALATE``: the listing is hidden and sent to
   urgent human review. Youth indicators never act on their own, so normal listings
   for children's clothes, toys or school books are unaffected.

2. **CSAM hash-matching integration point**: :class:`CsamHashMatcher` is the interface
   for an industry hash-matching service such as **Microsoft PhotoDNA** or
   **Thorn Safer**. Those services hold the hash lists of known child sexual abuse
   material; this project deliberately contains no such hashes, term lists or data.

   *Before launch* implement :class:`CsamHashMatcher` with your provider's SDK/API
   (credentials via environment variables, never in code) and register it with
   :func:`set_csam_matcher` at start-up. A match must return ``ESCALATE``, must
   **not** be shown to the seller as a reason, the content must be preserved as your
   provider/legal counsel instructs, and it must be reported through the legally
   required channels (e.g. NCMEC CyberTipline when your provider requires it, the
   Tanzania Police Force cyber-crime unit and TCRA under the Cybercrimes Act 2015).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class CsamMatchResult:
    """Outcome of a hash-matching lookup."""

    configured: bool
    matched: bool = False
    provider: str = "none"
    reference: Optional[str] = None  # provider match/report id, never shown to sellers


class CsamHashMatcher(Protocol):
    """Interface for an industry CSAM hash-matching service (PhotoDNA, Thorn Safer ...)."""

    def match(self, media_bytes: bytes, media_type: str) -> CsamMatchResult:
        """Look up one image/video.

        Parameters:
            media_bytes (bytes): Raw uploaded file bytes.
            media_type (str): ``"image"`` or ``"video"``.

        Returns:
            CsamMatchResult: Whether the media matched a known-CSAM hash.
        """
        ...


class NotConfiguredMatcher:
    """Placeholder used until a real provider is connected. Never matches."""

    def match(self, media_bytes: bytes, media_type: str) -> CsamMatchResult:
        """Return an unconfigured, non-matching result."""
        return CsamMatchResult(configured=False)


_matcher: CsamHashMatcher = NotConfiguredMatcher()


def set_csam_matcher(matcher: CsamHashMatcher) -> None:
    """Register the production hash-matching client (call once at start-up).

    Parameters:
        matcher (CsamHashMatcher): Provider implementation.
    """
    global _matcher
    _matcher = matcher
    logger.info("CSAM hash matcher registered: %s", type(matcher).__name__)


def csam_matcher_configured() -> bool:
    """True when a real provider (not the placeholder) is registered."""
    return not isinstance(_matcher, NotConfiguredMatcher)


def check_media(media_bytes: bytes, media_type: str) -> CsamMatchResult:
    """Run the registered matcher; errors are logged and treated as 'not matched'.

    Parameters:
        media_bytes (bytes): Raw media bytes.
        media_type (str): ``"image"`` or ``"video"``.

    Returns:
        CsamMatchResult: Lookup outcome.
    """
    try:
        return _matcher.match(media_bytes, media_type)
    except Exception as exc:
        logger.error("CSAM hash matcher failed (%s); media sent on without a hash check", exc)
        return CsamMatchResult(configured=True, matched=False, provider="error")
