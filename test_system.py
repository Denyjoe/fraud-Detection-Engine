"""End-to-End Automated Testing Suite.

Simulates and verifies:
1. Legitimate Kariakoo/Sinza clothing sale (returns ALLOW).
2. Disguised illegal listing ("Nauza gauni" with high mismatch/slang score at 3 AM -> returns BLOCK).
3. The live online learning loop: feeding actual_label=1 into /feedback and verifying
   that risk score increases on subsequent similar requests.
4. Layer 1 Swahili slang and illegal goods detection.
5. Escrow transaction scoring and audit logging verification.
"""

import base64
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from config.settings import DECISION_ALLOW, DECISION_BLOCK, DECISION_REVIEW
from main import app
from services.lexicon import get_lexicon


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI TestClient.

    Returns:
        TestClient: Synchronous test client.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_valid_image_base64() -> str:
    """Fixture generating a valid RGB flared dress image as base64 string.

    Returns:
        str: Base64 encoded JPEG image of kitenge women's dress on mannequin.
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    # Head and neck outline
    draw.ellipse([(98, 15), (126, 45)], fill=(220, 190, 170))
    draw.rectangle([(106, 45), (118, 60)], fill=(220, 190, 170))
    # Bodice
    draw.polygon([(92, 60), (132, 60), (124, 105), (100, 105)], fill=(180, 40, 90))
    # Flared dress skirt
    draw.polygon([(100, 105), (124, 105), (165, 210), (59, 210)], fill=(200, 50, 110))
    # African print / kitenge decorative patterns
    for y in range(115, 205, 15):
        draw.line([(70 + (y - 105) * 0.3, y), (154 - (y - 105) * 0.3, y)], fill=(240, 200, 50), width=3)
    # Belt / waist tie
    draw.rectangle([(98, 102), (126, 107)], fill=(240, 200, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def sample_charger_image_base64() -> str:
    """Fixture generating an electronics charger adapter image as base64 string.

    Returns:
        str: Base64 encoded JPEG image.
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(70, 70), (150, 150)], radius=15, fill=(40, 40, 40))
    draw.rectangle([(90, 40), (105, 70)], fill=(180, 180, 180))
    draw.rectangle([(115, 40), (130, 70)], fill=(180, 180, 180))
    draw.line([(110, 150), (110, 210)], fill=(60, 60, 60), width=6)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def sample_phone_image_base64() -> str:
    """Fixture generating a smartphone device image as base64 string.

    Returns:
        str: Base64 encoded JPEG image.
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(65, 25), (155, 195)], radius=12, fill=(20, 20, 20))
    draw.rounded_rectangle([(72, 35), (148, 185)], radius=8, fill=(70, 130, 220))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def sample_grocery_image_base64() -> str:
    """Fixture generating a grocery flour sack image as base64 string.

    Returns:
        str: Base64 encoded JPEG image.
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Sack top pleats
    draw.polygon([(90, 30), (134, 30), (120, 65), (104, 65)], fill=(225, 205, 170))
    # Tie rope
    draw.line([(96, 65), (128, 65)], fill=(100, 50, 20), width=5)
    # Sack bulging body
    draw.polygon([(104, 65), (120, 65), (170, 120), (165, 195), (59, 195), (54, 120)], fill=(230, 210, 175), outline=(180, 150, 110))
    # White flour label
    draw.rectangle([(75, 110), (149, 160)], fill=(255, 255, 255), outline=(180, 150, 110))
    # Flour spill at base
    draw.ellipse([(65, 190), (159, 215)], fill=(250, 250, 250))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def sample_cannabis_image_base64() -> str:
    """Fixture loading a real photo of cannabis buds (public Wikimedia image, see tests/assets/SOURCES.md).

    A real photo replaced the earlier hand-drawn leaf polygon: a clip-art drawing is
    (correctly) only sent to REVIEW, while this test is about a real contraband photo.

    Returns:
        str: Base64 encoded JPEG photo of cannabis buds.
    """
    from pathlib import Path

    photo = Path(__file__).resolve().parent / "tests" / "assets" / "cannabis.jpg"
    return base64.b64encode(photo.read_bytes()).decode("utf-8")


@pytest.fixture
def sample_suit_image_base64() -> str:
    """Fixture generating a men's formal suit silhouette as base64 string.

    Returns:
        str: Base64 encoded JPEG image of men's formal suit jacket, lapels, shirt, and tie.
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    # Dark charcoal suit jacket
    draw.polygon([(40, 220), (50, 70), (100, 40), (124, 40), (174, 70), (184, 220)], fill=(25, 28, 36))
    # White dress shirt collar inside
    draw.polygon([(100, 40), (124, 40), (112, 110)], fill=(255, 255, 255))
    # Red necktie
    draw.polygon([(108, 60), (116, 60), (118, 170), (112, 190), (106, 170)], fill=(180, 20, 20))
    # Suit lapels
    draw.polygon([(50, 70), (95, 130), (105, 45)], fill=(35, 40, 52))
    draw.polygon([(174, 70), (129, 130), (119, 45)], fill=(35, 40, 52))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@pytest.fixture
def sample_screenshot_image_base64() -> str:
    """Fixture generating a mobile phone screen capture as base64 string.

    Returns:
        str: Base64 encoded JPEG image of a mobile phone digital screenshot.
    """
    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), color=(240, 242, 245))
    draw = ImageDraw.Draw(img)
    # Status bar top
    draw.rectangle([(0, 0), (224, 24)], fill=(18, 140, 126))
    draw.text((10, 5), "12:30 PM", fill=(255, 255, 255))
    # App header
    draw.rectangle([(0, 24), (224, 60)], fill=(7, 94, 84))
    draw.text((15, 35), "WhatsApp Chat - Order #481", fill=(255, 255, 255))
    # Chat bubbles
    draw.rounded_rectangle([(15, 75), (160, 110)], radius=8, fill=(255, 255, 255))
    draw.text((22, 85), "Tuma pesa kwenye namba hii", fill=(30, 30, 30))
    draw.rounded_rectangle([(60, 125), (210, 160)], radius=8, fill=(220, 248, 198))
    draw.text((70, 135), "Nimetuma 450,000 TZS", fill=(30, 30, 30))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_health_endpoint(client: TestClient) -> None:
    """Verify health check returns healthy status and platform metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["currency"] == "TZS"


def test_layer1_swahili_slang_detection() -> None:
    """Verify the Layer 1 lexicon detects Swahili illegal goods and coded slang."""
    lexicon = get_lexicon()

    def analyse(text: str):
        result = lexicon.analyse(text)
        worst = result.worst()
        terms = {t for c in result.categories.values() for t in c.terms}
        return (worst.score if worst else 0.0), terms, result.categories

    # Test 1: Narcotics (Bangi & Cha Arusha)
    score, matched, categories = analyse("Nauza mzigo safi wa cha arusha na bangi freshi kwa bei poa")
    assert score >= 0.85
    assert "bangi" in matched
    assert "cha arusha" in matched
    assert categories["DRUGS"].level == DECISION_BLOCK

    # Test 2: Firearms (Bastola)
    score_weapon, matched_weapon, categories_weapon = analyse("Kuna bastola ndogo na risasi zinapatikana Mwenge")
    assert score_weapon >= 0.90
    assert "bastola" in matched_weapon
    assert "risasi" in matched_weapon
    assert categories_weapon["WEAPONS"].level == DECISION_BLOCK

    # Test 3: Financial Scam (Upatu / Pesa mara mbili)
    score_scam, matched_scam, _ = analyse("Jiunge na upatu upate pesa mara mbili ndani ya masaa 24 tuma kwa namba hii")
    assert score_scam >= 0.80
    assert "upatu" in matched_scam
    assert "pesa mara mbili" in matched_scam

    # Test 4: Benign text (Gauni la kitenge Kariakoo)
    score_clean, matched_clean, _ = analyse("Gauni zuri la kitenge pamba safi kutoka Kariakoo, rangi ya bluu")
    assert score_clean == 0.0
    assert len(matched_clean) == 0

    # Test 5: Context Disambiguation - Unga wa Sembe vs Narcotics
    score_flour, matched_flour, _ = analyse("Nauza unga wa sembe kilo 5 safi kabisa wa Dona kwa bei poa")
    assert score_flour == 0.0
    assert len(matched_flour) == 0

    # Test 6: Context Disambiguation - Kisu cha Jikoni vs Combat Weapon
    score_knife, matched_knife, _ = analyse("Seti ya visu vya jikoni na kisu cha nyama vya chuma cha pua")
    assert score_knife == 0.0
    assert len(matched_knife) == 0

    # Test 7: Context Disambiguation - Majani ya Chai vs Cannabis
    score_tea, matched_tea, _ = analyse("Majani ya chai ya kijani kutoka milima ya Usambara Tanga")
    assert score_tea == 0.0
    assert len(matched_tea) == 0

    # Test 8: Real Contraband Unga (Narcotics context)
    score_drug_unga, matched_drug_unga, _ = analyse("Kuna mzigo wa unga mweupe na kete za ngada")
    assert score_drug_unga >= 0.85
    assert any("unga" in m for m in matched_drug_unga)


def test_scenario_1_legitimate_kariakoo_clothing_sale(
    client: TestClient, sample_valid_image_base64: str
) -> None:
    """Requirement Test 1: Simulating a legitimate Kariakoo/Sinza clothing sale.

    Expected Result: ALLOW (probability < 0.40).
    """
    payload = {
        "listing_id": "LIST-KARIAKOO-1001",
        "seller_id": "USR-SINZA-FASHION",
        "title": "Gauni Nzuri ya Vitenge vya Kisasa Kariakoo",
        "description": "Nguo bora ya pamba, rangi ya kuvutia, inafaa kwa harusi na sherehe. Delivery ipo.",
        "price_tzs": 45000.0,
        "account_age_days": 180.0,
        "seller_completed_orders": 85,
        "seller_dispute_rate": 0.01,
        "image_base64": sample_valid_image_base64,
        "hour_of_day": 14,
        "is_repeat_buyer_same_day": 0,
    }

    response = client.post("/api/v1/moderate-listing", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["listing_id"] == "LIST-KARIAKOO-1001"
    assert data["decision"] == DECISION_ALLOW, f"Expected ALLOW but got {data['decision']}"
    assert data["fraud_probability"] < 0.40, f"Expected prob < 0.40, got {data['fraud_probability']}"
    assert data["layer1_details"]["swahili_slang_risk_score"] == 0.0
    assert data["audit_log_id"] > 0


def test_scenario_2_disguised_illegal_listing_block(
    client: TestClient, sample_cannabis_image_base64: str
) -> None:
    """Requirement Test 2: Simulating disguised illegal listing.

    Title says "Nauza gauni" but description contains coded contraband slang (bangi / unga / bastola),
    at 3:00 AM from a 1-day-old account with a real cannabis leaf photo.
    Expected Result: BLOCK (probability > 0.75).
    """
    payload = {
        "listing_id": "LIST-DISGUISED-9999",
        "seller_id": "USR-SHADY-007",
        "title": "Nauza gauni jipya la bei poa sana",
        "description": "Mzigo mzito wa cha arusha na msokoto kete za kutosha, wasiliana haraka kwa namba ya siri",
        "price_tzs": 650000.0,
        "account_age_days": 1.0,
        "seller_completed_orders": 0,
        "seller_dispute_rate": 0.35,
        "image_base64": sample_cannabis_image_base64,
        "hour_of_day": 3,  # 3:00 AM
        "is_repeat_buyer_same_day": 1,
    }

    response = client.post("/api/v1/moderate-listing", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["listing_id"] == "LIST-DISGUISED-9999"
    assert data["decision"] == DECISION_BLOCK, f"Expected BLOCK but got {data['decision']}"
    assert data["fraud_probability"] > 0.75, f"Expected prob > 0.75, got {data['fraud_probability']}"
    assert len(data["risk_reasons"]) > 0
    assert any("cha arusha" in reason or "msokoto" in reason or "slang" in reason.lower() for reason in data["risk_reasons"])


def test_real_clip_cannabis_image_with_innocent_gauni_title_blocked(
    client: TestClient, sample_cannabis_image_base64: str
) -> None:
    """Critical Bug Verification Test:

    When an image of cannabis/marijuana is uploaded under the innocent title "Gauni la harusi"
    with zero contraband text in the description, zero disputes, and a daytime transaction,
    the real HuggingFace CLIP model must detect the prohibited visual content, trigger
    visual_contraband_detected=True, and force decision=BLOCK (> 90% risk).
    """
    payload = {
        "listing_id": "LIST-CONTRABAND-CLIP-01",
        "seller_id": "USR-DISGUISED-BOT-88",
        "title": "Gauni Nzuri ya Harusi Sinza Kariakoo",
        "description": "Gauni safi kabisa ya pamba rangi nyeupe kwa ajili ya maharusi Sinza Dar es Salaam.",
        "price_tzs": 45000.0,
        "category": "fashion",
        "account_age_days": 60.0,
        "seller_completed_orders": 12,
        "seller_dispute_rate": 0.0,
        "image_base64": sample_cannabis_image_base64,
        "hour_of_day": 14,
        "is_repeat_buyer_same_day": 0,
    }

    response = client.post("/api/v1/moderate-listing", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["listing_id"] == "LIST-CONTRABAND-CLIP-01"
    assert data["decision"] == DECISION_BLOCK, f"Expected BLOCK but got {data['decision']}"
    assert data["fraud_probability"] >= 0.90, f"Expected prob >= 0.90, got {data['fraud_probability']}"
    assert data["visual_contraband_detected"] is True, "Expected visual_contraband_detected to be True"
    assert "cannabis" in data["top_visual_label"].lower(), f"Expected cannabis concept, got {data['top_visual_label']}"
    assert data["content_decision"] == DECISION_BLOCK
    assert any("photo appears to show illegal drugs" in reason for reason in data["risk_reasons"]), (
        f"Expected visual contraband reason in {data['risk_reasons']}"
    )


def test_scenario_3_live_online_learning_feedback_loop(client: TestClient) -> None:
    """Requirement Test 3: Simulating live online learning loop.

    1. We evaluate an ambiguous borderline transaction that starts with an intermediate score.
    2. We feed `actual_label=1` (Fraud) into the `/feedback` endpoint.
    3. We verify that `.learn_one()` updates the model weights and increases the fraud probability
       on subsequent identical or similar transactions.
    """
    borderline_features = {
        "price_tzs": 120000.0,
        "hour_of_day": 13,
        "account_age_days": 45.0,
        "seller_completed_orders": 8,
        "seller_dispute_rate": 0.06,
        "image_text_mismatch_score": 0.30,
        "swahili_slang_risk_score": 0.15,
        "is_repeat_buyer_same_day": 0,
    }

    tx_payload = {
        "transaction_id": "TX-FEEDBACK-TEST-001",
        "seller_id": "USR-BORDERLINE-42",
        "buyer_id": "USR-BUYER-88",
        **borderline_features,
    }

    # Step 1: Initial prediction
    initial_res = client.post("/api/v1/predict-transaction", json=tx_payload)
    assert initial_res.status_code == 200
    initial_prob = initial_res.json()["fraud_probability"]

    # Step 2: Feed multiple confirmed fraud reports for this behavioral pattern
    # In online gradient descent, providing feedback shifts weights towards fraud
    feedback_payload = {
        "transaction_id": "TX-FEEDBACK-TEST-001",
        "actual_label": 1,
        "features": borderline_features,
        "comment": "Escrow buyer reported fraud / account compromised",
    }

    for _ in range(5):
        feedback_res = client.post("/api/v1/feedback", json=feedback_payload)
        assert feedback_res.status_code == 200

    # Step 3: Verify subsequent evaluation produces a strictly higher risk score
    follow_up_payload = {
        "transaction_id": "TX-FEEDBACK-TEST-002",
        "seller_id": "USR-BORDERLINE-42",
        "buyer_id": "USR-BUYER-88",
        **borderline_features,
    }
    follow_up_res = client.post("/api/v1/predict-transaction", json=follow_up_payload)
    assert follow_up_res.status_code == 200
    updated_prob = follow_up_res.json()["fraud_probability"]

    assert updated_prob > initial_prob, (
        f"Expected online model probability to increase after fraud feedback. "
        f"Initial: {initial_prob:.4f}, After Feedback: {updated_prob:.4f}"
    )


def test_multipart_file_upload_listing(client: TestClient) -> None:
    """Verify multipart/form-data upload with binary image file."""
    from PIL import ImageDraw

    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(70, 70), (150, 150)], radius=15, fill=(40, 40, 40))
    draw.rectangle([(90, 40), (105, 70)], fill=(180, 180, 180))
    draw.rectangle([(115, 40), (130, 70)], fill=(180, 180, 180))
    draw.line([(110, 150), (110, 210)], fill=(60, 60, 60), width=6)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    files = {"image": ("charger.png", buffer.getvalue(), "image/png")}
    data = {
        "listing_id": "LIST-UPLOAD-4401",
        "seller_id": "USR-MWENGE-01",
        "title": "Chaja Original ya Simu Aina ya Type C",
        "description": "Fast charging chaja mpya kabisa kutoka dukani Mwenge.",
        "price_tzs": "35000",
        "category": "electronics",
        "account_age_days": "120",
        "seller_completed_orders": "40",
        "seller_dispute_rate": "0.02",
        "hour_of_day": "15",
        "is_repeat_buyer_same_day": "0",
    }

    response = client.post("/api/v1/moderate-listing-upload", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["listing_id"] == "LIST-UPLOAD-4401"
    assert res_data["decision"] in (DECISION_ALLOW, DECISION_REVIEW)


def test_validation_errors(client: TestClient) -> None:
    """Verify that malformed requests return HTTP 422 Unprocessable Entity."""
    # Invalid negative price and empty title
    invalid_payload = {
        "listing_id": "INVALID-1",
        "seller_id": "USR-1",
        "title": "   ",  # Blank title
        "description": "Valid description",
        "price_tzs": -5000.0,  # Negative price
    }
    response = client.post("/api/v1/moderate-listing", json=invalid_payload)
    assert response.status_code == 422


def test_metrics_and_audit_logging(client: TestClient) -> None:
    """Verify that system metrics endpoint accurately reflects database logs."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    metrics = response.json()

    assert "total_inferences" in metrics
    assert "total_feedbacks" in metrics
    assert "running_accuracy" in metrics
    assert metrics["total_inferences"] > 0
    assert metrics["total_feedbacks"] > 0


def test_serve_workbench_ui(client: TestClient) -> None:
    """Verify that root GET / serves the interactive Testing Workbench UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "pamostore fraud check" in response.text
    assert "Live Online Learning" in response.text


def test_audit_stats_endpoint(client: TestClient) -> None:
    """Verify that GET /api/v1/audit-stats returns detailed ledger and telemetry."""
    response = client.get("/api/v1/audit-stats")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "recent_logs" in data
    assert isinstance(data["recent_logs"], list)
    assert data["active_currency"] == "TZS"


def test_legitimate_mwenge_charger_and_unga_wa_sembe_allow(
    client: TestClient,
    sample_charger_image_base64: str,
    sample_grocery_image_base64: str,
) -> None:
    """Requirement Test: Verify that 35,000 TZS Type-C charger and unga wa sembe 5kg return ALLOW (< 35% risk)."""
    # 1. 35,000 TZS Type-C fast charger from Mwenge
    charger_payload = {
        "listing_id": "LIST-MWENGE-CHARGER-01",
        "seller_id": "USR-MWENGE-ELECTRONICS",
        "title": "Chaja Original ya Simu Fast Charging Type-C",
        "description": "Fast charger Type-C mpya kabisa dukani Mwenge, inachaji simu zote za kisasa haraka.",
        "price_tzs": 35000.0,
        "category": "electronics",
        "account_age_days": 90.0,
        "seller_completed_orders": 45,
        "seller_dispute_rate": 0.01,
        "image_base64": sample_charger_image_base64,
        "hour_of_day": 14,
        "is_repeat_buyer_same_day": 0,
    }
    res_charger = client.post("/api/v1/moderate-listing", json=charger_payload)
    assert res_charger.status_code == 200
    charger_data = res_charger.json()
    assert charger_data["category"] == "electronics"
    assert charger_data["decision"] == DECISION_ALLOW
    assert charger_data["fraud_probability"] < 0.35, f"Charger risk was {charger_data['fraud_probability']}"

    # 2. Unga wa sembe 5kg
    flour_payload = {
        "listing_id": "LIST-KARI-SEMBE-02",
        "seller_id": "USR-GROCERY-STORE",
        "title": "Unga wa Sembe Bora Kilo 5",
        "description": "Unga safi wa sembe namba moja kwa ugali laini na mtamu, kilo 5 kwa bei ya jumla.",
        "price_tzs": 15000.0,
        "category": "groceries",
        "account_age_days": 120.0,
        "seller_completed_orders": 60,
        "seller_dispute_rate": 0.0,
        "image_base64": sample_grocery_image_base64,
        "hour_of_day": 11,
        "is_repeat_buyer_same_day": 0,
    }
    res_flour = client.post("/api/v1/moderate-listing", json=flour_payload)
    assert res_flour.status_code == 200
    flour_data = res_flour.json()
    assert flour_data["category"] == "groceries"
    assert flour_data["layer1_details"]["swahili_slang_risk_score"] == 0.0
    assert flour_data["decision"] == DECISION_ALLOW, flour_data["risk_reasons"]
    assert flour_data["fraud_probability"] < 0.35, f"Flour risk was {flour_data['fraud_probability']}"


def test_rule_b_expensive_night_item_not_blocked(
    client: TestClient, sample_phone_image_base64: str
) -> None:
    """Requirement Test (Rule B): A legitimate seller listing an expensive item late at night

    with matching product image and zero disputes must never be auto-blocked.
    """
    expensive_phone_payload = {
        "listing_id": "LIST-EXP-PHONE-03",
        "seller_id": "USR-ESTABLISHED-SELLER",
        "title": "Smartphone Original Mpya Dukani",
        "description": "Simu mpya kabisa haijafunguliwa, risiti na waranti ipo miaka miwili.",
        "price_tzs": 1800000.0,
        "category": "electronics",
        "account_age_days": 150.0,
        "seller_completed_orders": 30,
        "seller_dispute_rate": 0.0,
        "image_base64": sample_phone_image_base64,
        "hour_of_day": 2,  # 2:00 AM
        "is_repeat_buyer_same_day": 0,
    }
    response = client.post("/api/v1/moderate-listing", json=expensive_phone_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] in (DECISION_ALLOW, DECISION_REVIEW)
    assert data["decision"] != DECISION_BLOCK


@pytest.mark.parametrize("test_price", [100.0, 500.0, 900.0, 1450.0, 1450.50])
def test_flexible_price_amounts_accepted(
    client: TestClient, sample_valid_image_base64: str, test_price: float
) -> None:
    """Verify that any positive price in TZS (including small and decimal amounts)

    is accepted by the API without 422 validation errors.
    """
    payload = {
        "listing_id": f"LIST-PRICE-TEST-{int(test_price)}",
        "seller_id": "USR-PRICE-TEST",
        "title": "Gauni Nzuri ya Pamba",
        "description": "Nguo nzuri ya pamba ya kisasa, rangi ya bluu.",
        "price_tzs": test_price,
        "category": "fashion",
        "account_age_days": 60.0,
        "seller_completed_orders": 10,
        "seller_dispute_rate": 0.0,
        "image_base64": sample_valid_image_base64,
        "hour_of_day": 14,
        "is_repeat_buyer_same_day": 0,
    }
    response = client.post("/api/v1/moderate-listing", json=payload)
    assert response.status_code == 200, f"Failed for price {test_price}: {response.text}"
    data = response.json()
    assert data["decision"] == DECISION_ALLOW


def test_suit_image_uploaded_for_gauni_listing_triggers_mismatch(
    client: TestClient, sample_suit_image_base64: str
) -> None:
    """Fine-Grained Mismatch Test: Uploading a men's suit photo under 'Gauni Nzuri ya Vitenge'

    must trigger a severe mismatch score (>= 0.75), comparison status MISMATCH, and HOLD/BLOCK.
    """
    payload = {
        "listing_id": "LIST-SUIT-FOR-DRESS-99",
        "seller_id": "USR-MISMATCH-TEST",
        "title": "Gauni Nzuri ya Vitenge vya Kisasa",
        "description": "Gauni safi ya kike ya kitenge cha rangi, inafaa sana kwa harusi na sherehe Sinza.",
        "price_tzs": 45000.0,
        "category": "fashion",
        "account_age_days": 15.0,
        "seller_completed_orders": 2,
        "seller_dispute_rate": 0.0,
        "image_base64": sample_suit_image_base64,
        "hour_of_day": 14,
        "is_repeat_buyer_same_day": 0,
    }
    response = client.post("/api/v1/moderate-listing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["visual_comparison_status"] == "MISMATCH"
    assert data["layer1_details"]["image_text_mismatch_score"] >= 0.75
    assert data["decision"] in (DECISION_REVIEW, DECISION_BLOCK)
    assert any("does not match the title" in r for r in data["risk_reasons"])


def test_screenshot_uploaded_triggers_non_product_mismatch(
    client: TestClient, sample_screenshot_image_base64: str
) -> None:
    """Non-Product Fraud Test: Uploading a digital screenshot instead of physical merchandise

    must be classified as non-product / screenshot with high mismatch score.
    """
    payload = {
        "listing_id": "LIST-SCREENSHOT-TEST-01",
        "seller_id": "USR-SCREENSHOT-SELLER",
        "title": "Gauni Nzuri ya Harusi",
        "description": "Nguo ya harusi, nicheki WhatsApp kwenye namba ya screenshot.",
        "price_tzs": 75000.0,
        "category": "fashion",
        "account_age_days": 5.0,
        "seller_completed_orders": 0,
        "seller_dispute_rate": 0.0,
        "image_base64": sample_screenshot_image_base64,
        "hour_of_day": 15,
        "is_repeat_buyer_same_day": 0,
    }
    response = client.post("/api/v1/moderate-listing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["visual_comparison_status"] == "MISMATCH"
    assert data["layer1_details"]["image_text_mismatch_score"] >= 0.75
    assert data["decision"] in (DECISION_REVIEW, DECISION_BLOCK)
    assert any("screenshot/document" in r or "does not match the title" in r for r in data["risk_reasons"])


def test_matching_garment_returns_visual_match(
    client: TestClient, sample_valid_image_base64: str
) -> None:
    """Positive Match Test: Legitimate gown photo for 'Gauni Nzuri'

    must return visual_comparison_status MATCH and low mismatch score <= 0.25.
    """
    payload = {
        "listing_id": "LIST-MATCHING-DRESS-01",
        "seller_id": "USR-MATCHING-SELLER",
        "title": "Gauni Nzuri ya Vitenge Sinza",
        "description": "Gauni ya kitenge pamba safi, saizi zote zipo.",
        "price_tzs": 45000.0,
        "category": "fashion",
        "account_age_days": 90.0,
        "seller_completed_orders": 25,
        "seller_dispute_rate": 0.0,
        "image_base64": sample_valid_image_base64,
        "hour_of_day": 14,
        "is_repeat_buyer_same_day": 0,
    }
    response = client.post("/api/v1/moderate-listing", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["visual_comparison_status"] == "MATCH"
    assert data["layer1_details"]["image_text_mismatch_score"] <= 0.25
    assert data["decision"] == DECISION_ALLOW




