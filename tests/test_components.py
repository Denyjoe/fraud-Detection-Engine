"""Unit tests for the text normaliser, lexicon, contact detector, fusion and child-safety rules."""

import pytest

from config.settings import DECISION_ALLOW, DECISION_BLOCK, DECISION_ESCALATE, DECISION_REVIEW
from services import child_safety
from services.contact_detector import detect_contacts
from services.fusion import CAT_YOUTH, FEATURE_NAMES, Signal, decide
from services.lexicon import Lexicon, get_lexicon
from services.text_normalizer import normalise_variants

FEATURES = {name: 0.0 for name in FEATURE_NAMES}


@pytest.mark.parametrize("text", ["b4ng1", "B.A.N.G.I", "b a n g i", "baaangi", "b-a-n-g-i", "BANGI", "ba​ngi",
                                  "bаngi"])
def test_obfuscated_bangi_is_blocked(text: str) -> None:
    """Leetspeak, spacing, repeated letters, zero-width and Cyrillic homoglyphs are decoded."""
    worst = get_lexicon().analyse(f"Nauza {text} safi").worst()
    assert worst is not None and worst.category == "DRUGS" and worst.level == DECISION_BLOCK


def test_plain_variant_keeps_prices_and_numbers() -> None:
    """Leetspeak decoding never touches the plain variant (prices, phone numbers)."""
    kind, plain = normalise_variants("Bei 15000 TZS, 0712 345 678")[0]
    assert kind == "plain" and "15000" in plain and "0712 345 678" in plain


@pytest.mark.parametrize("text", [
    "Unga wa sembe 5kg", "Unga safi wa sembe kwa ugali", "majani ya chai Lipton", "Glue gun 40W",
    "Silencer ya gari Toyota", "Ivory colour dress", "Kobe Bryant shoes", "Massage chair", "Heroine movie",
    "Piston rings", "kisu cha jikoni", "Kalamu za risasi", "Bangili za dhahabu", "Ford Escort 1998",
    "Gongo la Mboto plot", "weed killer", "joint pain cream", "iPhone iCloud clean IMEI clean",
])
def test_false_positive_protections(text: str) -> None:
    """Legitimate phrases from the allow lists never reach REVIEW."""
    worst = get_lexicon().analyse(text).worst()
    assert worst is None or worst.score < 0.35, f"{text!r} -> {worst.category} {worst.level} {worst.terms}"


def test_typo_matching_reviews() -> None:
    """A one-letter typo of a strong term is caught (as REVIEW) and excluded words are not."""
    assert get_lexicon().analyse("kokaine safi").worst().level == DECISION_REVIEW
    assert get_lexicon().analyse("heroine").worst() is None


def test_weak_word_needs_context() -> None:
    """Ambiguous words only count together with context words or emoji."""
    lex = get_lexicon()
    assert lex.analyse("majani makavu").worst().score < 0.35
    assert lex.analyse("majani ya kuvuta").worst().level == DECISION_BLOCK
    assert lex.analyse("majani 🍁 mazuri").worst().level == DECISION_BLOCK


def test_lexicon_is_editable_json(tmp_path) -> None:
    """New slang can be added through JSON without code changes."""
    data = {"categories": {"DRUGS": {"strong": ["zigzag mpya"], "weak_with_context": "BLOCK"}}}
    lex = Lexicon(data)
    assert lex.analyse("nauza zigzag mpya").worst().level == DECISION_BLOCK


@pytest.mark.parametrize("text,found", [
    ("Nicheki WhatsApp 0712 345 678", True),
    ("Piga +255-754-111-222", True),
    ("message me on WhatsApp, 0712, 345, 678.", True),
    ("Bei 1,250,000 TZS", False),
    ("sifuri saba moja mbili tatu nne tano sita saba", True),
    ("Lipa namba 123456", True),
    ("join telegram t.me/dealz", True),
    ("Samsung A15, WhatsApp inafanya kazi vizuri", False),
    ("Bei 255,000 TZS", False),
    ("IMEI 356938035643809", False),
])
def test_contact_detection(text: str, found: bool) -> None:
    """Off-platform contact attempts are detected; product features and prices are not."""
    assert detect_contacts(text).found is found


def _signal(category: str, level: str, source: str = "text") -> Signal:
    return Signal(source, category, level, 0.9 if level == DECISION_BLOCK else 0.6, "test")


def test_escalate_on_sexual_plus_youth() -> None:
    """Child safety: any sexual signal combined with a youth indicator escalates."""
    result = decide([_signal("SEXUAL", DECISION_REVIEW), Signal("text", CAT_YOUTH, "WEAK", 0.0, "youth")], FEATURES)
    assert result.decision == DECISION_ESCALATE
    assert "CHILD_SAFETY" in result.categories


def test_escalate_on_image_nsfw_plus_age_estimate() -> None:
    """Image nudity plus an age-estimator youth signal escalates even if the nudity alone would block."""
    result = decide([_signal("SEXUAL", DECISION_BLOCK, "image"), Signal("image", CAT_YOUTH, "WEAK", 0.6, "age")],
                    FEATURES)
    assert result.decision == DECISION_ESCALATE


def test_youth_alone_does_not_flag() -> None:
    """Youth words without a sexual signal never flag (children's clothes, toys, books)."""
    result = decide([Signal("text", CAT_YOUTH, "WEAK", 0.0, "youth")], FEATURES)
    assert result.decision == DECISION_ALLOW


def test_escalate_seller_message_does_not_reveal_reason() -> None:
    """The seller message for ESCALATE is neutral."""
    result = decide([_signal("SEXUAL", DECISION_REVIEW), Signal("text", CAT_YOUTH, "WEAK", 0.0, "y")], FEATURES)
    assert "mtoto" not in result.seller_message_sw.lower() and "child" not in result.seller_message_en.lower()


def test_ml_alone_cannot_block() -> None:
    """A text-ML-only signal is capped at REVIEW."""
    result = decide([Signal("text_ml", "DRUGS", DECISION_REVIEW, 0.6, "ml")], FEATURES)
    assert result.decision == DECISION_REVIEW


def test_csam_matcher_interface(monkeypatch) -> None:
    """A registered hash-matching provider is used and a match escalates the listing."""
    from services.moderation_layer1 import ContentModerationLayer1, MediaFile

    class FakeProvider:
        def match(self, media_bytes: bytes, media_type: str) -> child_safety.CsamMatchResult:
            return child_safety.CsamMatchResult(configured=True, matched=True, provider="fake", reference="r1")

    monkeypatch.setattr(child_safety, "_matcher", FakeProvider())
    result = ContentModerationLayer1().moderate("Picha", "", [MediaFile("x.jpg", b"\xff\xd8\xff\xe0 fake")],
                                                run_text_ml=False)
    assert result["decision"] == DECISION_ESCALATE
    assert result["checks"]["csam_hash_matching"] == "configured"


def test_csam_matcher_not_configured_by_default() -> None:
    """Out of the box the placeholder is active and reported as NOT configured."""
    assert child_safety.csam_matcher_configured() is False
