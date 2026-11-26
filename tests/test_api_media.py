"""API and media tests: POST /moderate, GET /health, video, hash blocklist, auth and limits."""

import io
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from config.settings import DECISION_ALLOW, DECISION_BLOCK, DECISION_REVIEW
from main import app

ASSETS = Path(__file__).resolve().parent / "assets"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


def _file(name: str, content_type: str = "image/jpeg") -> tuple:
    return ("files", (name, (ASSETS / name).read_bytes(), content_type))


def make_video(tmp_path: Path, images: List[str], seconds_each: int = 2) -> Path:
    """Build a small MP4 from test photos (frames of each photo for ``seconds_each`` seconds)."""
    import cv2

    path = tmp_path / "listing.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 320))
    for name in images:
        frame = cv2.cvtColor(np.array(Image.open(ASSETS / name).convert("RGB").resize((320, 320))), cv2.COLOR_RGB2BGR)
        for _ in range(5 * seconds_each):
            writer.write(frame)
    writer.release()
    return path


def test_health_reports_models(client: TestClient) -> None:
    """GET /health lists every model's status and the CSAM integration state."""
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert {"text_zero_shot", "clip", "nsfw_classifier", "nudenet", "ocr"} <= set(data["models"])
    assert data["csam_hash_matching"] == "NOT_CONFIGURED"


def test_moderate_text_only_allow(client: TestClient) -> None:
    """A clean Swahili listing is allowed with a Swahili seller message."""
    res = client.post("/moderate", data={"title": "Gauni la kitenge", "description": "Pamba safi saizi zote"})
    assert res.status_code == 200
    body = res.json()
    assert body["decision"] == DECISION_ALLOW
    assert body["seller_message_sw"].startswith("Tangazo lako limekubaliwa")


def test_moderate_photo_mismatch_blocks_contraband(client: TestClient) -> None:
    """Title says shoes, photo shows cannabis -> BLOCK with an explanation naming both."""
    res = client.post("/moderate", data={"title": "Raba nzuri za Nike", "description": "Size 42"},
                      files=[_file("cannabis.jpg")])
    body = res.json()
    assert body["decision"] == DECISION_BLOCK
    assert any(r["source"] == "image" and r["category"] == "DRUGS" and "claims" in r["detail"]
               for r in body["reasons"])


def test_moderate_multiple_files(client: TestClient) -> None:
    """Several files are all checked; one bad photo among good ones still blocks."""
    res = client.post("/moderate", data={"title": "Viatu vya kiume", "description": "Ngozi"},
                      files=[_file("sneakers.jpg"), _file("pistol.jpg")])
    body = res.json()
    assert body["decision"] == DECISION_BLOCK
    assert len(body["details"]["media"]) == 2
    assert any(r.get("file") == "pistol.jpg" for r in body["reasons"])


def test_moderate_video_frames(client: TestClient, tmp_path: Path) -> None:
    """A video whose later frames show a gun is blocked (frame sampling)."""
    video = make_video(tmp_path, ["sneakers.jpg", "pistol.jpg"])
    res = client.post("/moderate", data={"title": "Raba mpya", "description": "Video ya bidhaa"},
                      files=[("files", ("listing.mp4", video.read_bytes(), "video/mp4"))])
    body = res.json()
    assert body["decision"] == DECISION_BLOCK, body["reasons"]
    media = body["details"]["media"][0]
    assert media["type"] == "video" and media["frames_analysed"] >= 4


def test_unreadable_image_goes_to_review(client: TestClient) -> None:
    """Corrupt uploads are not silently accepted."""
    res = client.post("/moderate", data={"title": "Gauni"}, files=[("files", ("bad.jpg", b"not an image", "image/jpeg"))])
    assert res.json()["decision"] == DECISION_REVIEW


def test_upload_limits(client: TestClient, monkeypatch) -> None:
    """Too many files -> HTTP 413."""
    from api import routes

    monkeypatch.setitem(routes.API_CFG, "max_files", 1)
    res = client.post("/moderate", data={"title": "Gauni"}, files=[_file("sneakers.jpg"), _file("bicycle.jpg")])
    assert res.status_code == 413


def test_api_key_required_when_configured(client: TestClient, monkeypatch) -> None:
    """With MODERATION_API_KEY set, requests need the X-API-Key header."""
    monkeypatch.setenv("MODERATION_API_KEY", "test-secret")
    assert client.post("/moderate", data={"title": "Gauni"}).status_code == 401
    ok = client.post("/moderate", data={"title": "Gauni"}, headers={"X-API-Key": "test-secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200  # health stays public for load balancers


def test_hash_blocklist(tmp_path: Path, monkeypatch) -> None:
    """An image added to the pHash blocklist is blocked even after resizing."""
    from services import image_analyzer
    from services.moderation_layer1 import ContentModerationLayer1, MediaFile

    blocklist = tmp_path / "hashes.json"
    image = Image.open(ASSETS / "bicycle.jpg").convert("RGB")
    image_analyzer.add_to_blocklist(image, "DRUGS", path=blocklist)
    monkeypatch.setattr(image_analyzer, "HASH_BLOCKLIST_PATH", blocklist)

    buf = io.BytesIO()
    image.resize((300, 200)).save(buf, format="JPEG", quality=70)
    result = ContentModerationLayer1().moderate("Baiskeli", "", [MediaFile("re-upload.jpg", buf.getvalue())],
                                                run_text_ml=False)
    assert result["decision"] == DECISION_BLOCK
    assert any(r["source"] == "hash" for r in result["reasons"])
