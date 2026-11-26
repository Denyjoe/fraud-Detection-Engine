# pamostore fraud check  Content Moderation & Fraud Detection Engine

Moderation and fraud service for a **Tanzanian online store**. Every listing (title, description, photos, videos) gets one of four decisions, the reasons behind it, and a ready-to-show **Swahili message for the seller**:

| Decision | Meaning | What your store should do |
|---|---|---|
| `ALLOW` | Nothing found | Publish |
| `REVIEW` | Suspicious, or off-platform contact, or photo doesn't match the title | Hold for a moderator |
| `BLOCK` | Clearly prohibited (drugs, weapons, ivory, nudity, fake money …) | Reject; show `seller_message_sw` |
| `ESCALATE` | Child-safety signal | Hide immediately; **urgent** human review by your safety team |

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688)

## Contents
- [What it detects](#what-it-detects) · [How it works](#how-it-works)
- [Install](#install) · [Download models](#download-models) · [GPU](#gpu-nvidia)
- [CLI](#cli-usage) · [REST API](#rest-api) · [Java example](#java-example)
- [Adding slang](#adding-new-slang-no-code-changes) · [Banning an image](#banning-a-specific-image)
- [Train & evaluate](#training-and-evaluation) · [Tuning thresholds](#tuning-thresholds)
- [Child safety (mandatory before launch)](#child-safety-mandatory-before-launch)
- [Tests](#tests) · [Security](#security) · [Known limitations](#known-limitations) · [Project structure](#project-structure)

---

## What it detects

| Goal | How |
|---|---|
| Illegal goods in **Swahili or English**: drugs (bangi, mirungi, unga/kete, ngada…), weapons/ammo, wildlife trophies (pembe za ndovu, kakakuona, ngozi ya chui), fake money/documents/card fraud, stolen goods (IMEI change, iCloud bypass), sexual services, TZ-banned products (viroba, gongo, mkorogo, plastic bags, fake HIV cures, shisha), investment scams | Editable lexicon `config/lexicon.json` + multilingual zero-shot text model |
| Disguised spellings: `b4ng1`, `b.a.n.g.i`, `B A N G I`, `baaangi`, Cyrillic look-alike letters, invisible characters, typos (`kokaine`), emoji codes (🍁💨, 🔫, 🍆) | `services/text_normalizer.py` + typo matching + emoji context |
| Protects legit trade: *unga wa sembe*, *majani ya chai*, *glue gun*, *silencer ya gari*, *ivory colour*, *Kobe Bryant shoes*, *massage chair*, *piston*, *Heroine* (movie), *kalamu za risasi*, *Ford Escort*, *Gongo la Mboto*… | `allow` lists and `weak_cancel` words |
| **Pornographic images and videos** | Falconsai NSFW classifier + NudeNet detector (pretrained only); videos: sampled frames |
| **Photo doesn't match the product** ("shoes" with a photo of cannabis or a gun; "dress" with a photo of a phone) | CLIP zero-shot: contraband prompts + harmless look-alikes + product taxonomy + the seller's claim |
| **Text inside images** (OCR) checked with the same rules | EasyOCR |
| **Speech in videos** (optional) | Whisper transcription → same rules |
| **Off-platform contact**: TZ phone numbers (also spaced, or spelled out in Swahili), WhatsApp/Telegram invitations, links, "Lipa namba" | `services/contact_detector.py` |
| Re-uploads of already-banned images | Perceptual-hash blocklist `data/hash_blocklist.json` |
| Seller/price/escrow risk (your existing Layer 2) | River online model + guardrails |

## How it works

```
title + description ─► normaliser ─► lexicon (strong/review/weak+context/allow) ─┐
                    └► contact detector ─────────────────────────────────────────┤
                    └► zero-shot text model (mDeBERTa XNLI, Swahili-capable) ────┤
images ─► hash blocklist · CSAM hash hook · NSFW + NudeNet · CLIP · OCR ─► lexicon ┤──► fusion ─► ALLOW/REVIEW/BLOCK/ESCALATE
videos ─► sampled frames (same image checks) · optional Whisper audio ─► lexicon ─┤      + reasons + Swahili message
                       sexual signal + youth indicator ─► ESCALATE ──────────────┘
```

**Fusion rules** (`services/fusion.py`). Rules are authoritative, and the trained meta-model can only make a decision stricter:
1. CSAM hash match, or a sexual signal together with a youth indicator → `ESCALATE`.
2. Any BLOCK-level signal → `BLOCK`.
3. Meta-model probability ≥ its BLOCK threshold → `BLOCK`, but **only** if a non-ML signal also exists (ML alone never blocks); ≥ REVIEW threshold → `REVIEW`.
4. Any REVIEW-level signal, or several weak signals together → `REVIEW`. Otherwise `ALLOW`.

Every model loads **lazily** on first use and **fails softly**: if a model is missing or broken, that check switches off with a warning (shown in `GET /health` and in each response's `checks`) and everything else keeps working. If *both* nudity models are unavailable, listings with media go to `REVIEW` rather than passing unchecked (`decision.review_when_media_unchecked`).

---

## Install

Requirements: **Python 3.10+**, about **6 GB disk** (PyTorch and models), internet for the first run. Tested on Windows 11 with Python 3.14.

**Windows (PowerShell)**: one command. It creates `.venv`, installs the pinned requirements, trains the Layer 2 cold-start model and downloads the moderation models:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\run.ps1        # http://127.0.0.1:8000
```
**macOS / Linux**: `bash setup.sh` then `bash run.sh`.

Manual:
```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python generate_synthetic_data.py                       # Layer 2 cold-start model
python download_models.py                               # moderation models
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
Paths with spaces (like `C:\Users\Denis Wilson\...`) are fine. The console is switched to UTF-8 automatically so Swahili text and emoji print correctly.

### Download models

`python download_models.py` downloads every model enabled in `config/config.json` (about 2.5 GB, once) into the Hugging Face cache (set `HF_HOME` to move it). `--all` also fetches the optional ones (NLLB translation 2.4 GB, Whisper).

| Key in config.json | Default model (public) | Purpose | Default |
|---|---|---|---|
| `text_zero_shot` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | multilingual zero-shot text classification | on |
| `clip` | `openai/clip-vit-base-patch32` | contraband in photos, photo-vs-title mismatch | on |
| `nsfw_classifier` | `Falconsai/nsfw_image_detection` | nudity / porn | on |
| `nudenet` | NudeNet 3.4 (pip, ONNX) | exposed body-part detector | on |
| `ocr` | EasyOCR | text inside images | on |
| `age_estimator` | `nateraw/vit-age-classifier` | only runs when a sexual signal exists → `ESCALATE` | on |
| `translation` | `facebook/nllb-200-distilled-600M` | Swahili→English for CLIP prompts | off |
| `speech_to_text` | `openai/whisper-base` | video audio (also set `video.transcribe_audio`) | off |

After the first download you can run fully offline with `HF_HUB_OFFLINE=1`.

### GPU (NVIDIA)

The engine uses CUDA automatically when PyTorch can see a GPU (`"device": "auto"` in config.json; `"cpu"` forces CPU). The default `pip install torch` on Windows is **CPU-only**. For an NVIDIA GPU (e.g. RTX A2000) install the CUDA build first:
```powershell
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu128
```
On CPU, the text model and OCR add roughly 3–8 s per listing; a GPU brings the whole pipeline well under a second. `GET /health` shows the active `device`.

---

## CLI usage
```powershell
python moderate.py selftest                                     # sample listings + model status
python moderate.py check --title "Raba nzuri za Nike" --description "Size 42" --file "C:\Photos\raba 1.jpg"
python moderate.py check --title "Gauni" --file a.jpg --file b.jpg --file video.mp4 --json
python moderate.py block-image "C:\Banned\photo.jpg" --reason DRUGS
python moderate.py reload-lexicon                               # validate lexicon.json after editing
```

## REST API

Interactive docs: `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/moderate` | **Main endpoint for your backend.** Multipart: `title` (required), `description`, `files` (repeat for each image/video), optional `listing_id`, `seller_id`, `category`, `price_tzs` |
| `GET` | `/health` | Service status, device, per-model status, meta-model, CSAM integration |
| `POST` | `/api/v1/moderate-listing` | Listing + seller history (JSON, base64 image) → content checks + Layer 2 risk |
| `POST` | `/api/v1/moderate-listing-upload` | Same, multipart (`image` and/or `files`) |
| `POST` | `/api/v1/predict-transaction` | Escrow payment risk |
| `POST` | `/api/v1/feedback` | Ground truth for the online Layer 2 model |
| `GET` | `/api/v1/metrics`, `/api/v1/audit-stats` | Totals and the audit ledger |

```bash
curl -X POST http://127.0.0.1:8000/moderate \
  -H "X-API-Key: $MODERATION_API_KEY" \
  -F "title=Raba nzuri za Nike" -F "description=Size 42" \
  -F "files=@tests/assets/cannabis.jpg"
```
Response (abridged):
```json
{
  "decision": "BLOCK",
  "score": 0.99,
  "categories": ["DRUGS"],
  "reasons": [{"source": "image", "category": "DRUGS", "level": "BLOCK", "score": 0.99,
               "detail": "photo appears to show illegal drugs (99%) although the listing claims 'sneakers, athletic running shoes, or trainers'",
               "file": "cannabis.jpg"}],
  "seller_message_sw": "Tangazo lako limezuiwa kwa sababu picha au video ina maudhui yasiyoruhusiwa. Tafadhali weka picha halisi na safi ya bidhaa.",
  "seller_message_en": "Your listing was blocked because a photo or video contains prohibited content.",
  "category": "fashion",
  "checks": {"clip": "ready", "nsfw_classifier": "ready", "csam_hash_matching": "NOT_CONFIGURED", "...": "..."},
  "processing_ms": 2073,
  "audit_log_id": 17,
  "details": {"media": ["per-file scores, OCR text, pHash ..."], "zero_shot": null, "features": {"...": 0}}
}
```
HTTP errors: `401` bad/missing API key, `413` too many / too large files (`api.max_files`, `api.max_file_mb`), `422` invalid input, `500` internal error. Treat any error as `REVIEW` on your side.

### Java example

`examples/java/ModerationClient.java` is a dependency-free Java 11+ client (multipart upload, API key, health check):
```powershell
cd examples\java
javac ModerationClient.java
$env:MODERATION_URL="http://127.0.0.1:8000"; java ModerationClient "Raba nzuri za Nike" "Size 42" "..\..\tests\assets\sneakers.jpg"
```
The core call:
```java
ModerationClient client = new ModerationClient("http://127.0.0.1:8000", System.getenv("MODERATION_API_KEY"));
String json = client.moderate(Map.of("title", title, "description", description, "seller_id", sellerId), photoPaths);
// parse "decision": ALLOW -> publish, REVIEW -> moderator queue, BLOCK -> reject + seller_message_sw,
// ESCALATE -> hide now and page the safety team
```

---

## Adding new slang (no code changes)

Edit **`config/lexicon.json`** and restart the server (`python moderate.py reload-lexicon` checks the file first). Per category:

| List | Effect | Example |
|---|---|---|
| `strong` | one hit → **BLOCK** | `"bangi"`, `"pembe za ndovu"` |
| `review` | one hit → **REVIEW** (with a context word → `review_with_context`) | `"tramadol"`, `"upatu"` |
| `weak` | ambiguous; only counts with a `context` word or `emoji` of the same category (→ `weak_with_context`) | `"unga"`, `"majani"`, `"gun"` |
| `context` | words that make weak words suspicious | `"kuvuta"`, `"kunusa"`, `"bila leseni"` |
| `weak_cancel` | innocent context; a lone weak word is ignored | `"sembe"`, `"ugali"`, `"jikoni"` |
| `emoji` | emoji context (never blocks alone) | `"🍁"`, `"🔫"` |
| `global_allow` | phrases removed before matching | `"unga wa sembe"`, `"glue gun"`, `"kobe bryant"` |
| `fuzzy_exclude` | real words that must not be typo-matched | `"heroine"`, `"bangili"` |

Write terms in plain lower case. Obfuscated forms (`b4ng1`, `b.a.n.g.i`, `baaangi`) and one-letter typos of single-word `strong` terms are handled automatically. Seller messages are in `seller_messages`. If a normal listing gets flagged, add its phrase to `global_allow` or a word to `weak_cancel`, then add the case to `tests/data/test_listings.csv` so it stays fixed.

## Banning a specific image
`python moderate.py block-image path\to\image.jpg --reason DRUGS` stores its perceptual hash in `data/hash_blocklist.json`; re-uploads (resized or re-compressed) are then blocked.

---

## Training and evaluation

Labelled CSV, UTF-8 (`data/train_listings.csv` is an example with 128 rows, separate from the test set):
```
id,title,description,images,videos,label
L01,Unga wa sembe 5kg,Unga safi wa sembe kwa ugali,,,ALLOW
V07,Raba nzuri za Nike,Size 42,photos/v07_1.jpg;photos/v07_2.jpg,,BLOCK
R01,Gauni la kitenge,Nicheki WhatsApp 0712 345 678,,,REVIEW
```
`label` = the decision you expect: `ALLOW` (legit) or `REVIEW` / `BLOCK` / `ESCALATE`. Several files are separated by `;`, and paths are relative to the project folder.

```powershell
python -m training.train_meta_model train data/train_listings.csv            # -> models/meta_model.joblib
python -m training.train_meta_model evaluate tests/data/test_listings.csv --report report.json
```
* **train** runs each listing through the full engine to get its feature vector (cached in `data/feature_cache.json`), fits a logistic regression with sigmoid calibration, and chooses thresholds from out-of-fold predictions: BLOCK threshold for precision ≥ 0.95 (`--block-precision`), REVIEW threshold for recall ≥ 0.95 (`--review-recall`). The server picks the new model up on restart; `/health` → `meta_model` shows it.
* **evaluate** prints precision / recall / F1 (violation vs legit, and for BLOCK), the confusion matrix and the worst mistakes (false positives, false negatives, BLOCK-labelled listings only reviewed).

## Tuning thresholds

All in `config/config.json` (restart after editing):

| Setting | Default | Raise it to… / lower it to… |
|---|---|---|
| `image.nsfw_block_threshold`, `nudenet_block_threshold` | 0.85 / 0.60 | raise → fewer nudity blocks (more go to REVIEW) |
| `image.nsfw_review_threshold`, `nudenet_review_threshold` | 0.55 / 0.40 | lower → catch more borderline images |
| `image.clip_contraband_block_threshold` / `_review_threshold` | 0.75 / 0.50 | photo-of-drugs/guns/ivory confidence |
| `image.clip_mismatch_threshold`, `clip_mismatch_max_claimed` | 0.70 / 0.10 | photo-vs-title mismatch strictness |
| `text.zero_shot_review_threshold`, `zero_shot_max_normal` | 0.95 / 0.15 | when the text model alone sends to REVIEW |
| `decision.review_score` | 0.35 | several weak signals together → REVIEW |
| `decision.meta_min_review_threshold`, `meta_min_block_threshold` | 0.50 / 0.85 | safety floors: trained meta-model thresholds are never used below these (a small training set produced 0.29/0.32 in testing and over-flagged faux-leather listings) |
| `video.frames_to_sample`, `transcribe_audio` | 8 / false | video depth vs speed |
| `api.max_files`, `max_file_mb`, `cors_origins` | 10 / 25 / [] | upload limits; browser origins allowed to call the API |

Measure every change with `evaluate` on your own labelled listings before deploying it.

---

## Child safety (mandatory before launch)

* **Escalation rule (built in):** any sexual signal (text, image or video) combined with any youth indicator (ordinary age words such as *mtoto*, *mwanafunzi*, *under 18*, ages under 18, or the age estimator on an image that already has a sexual signal) returns **`ESCALATE`**. Youth words never flag anything on their own, so children's clothes, toys and school books are not affected. The seller message for `ESCALATE` is neutral and does not reveal the reason.
* **CSAM hash matching: you must connect it before launch.** `services/child_safety.py` defines the `CsamHashMatcher` interface for an industry hash-matching service such as **Microsoft PhotoDNA** or **Thorn Safer**. These providers hold the hash lists of known child sexual abuse material. This project deliberately contains **no** CSAM-related term lists, hashes or data. Implement `match(media_bytes, media_type)` with your provider's SDK/API (credentials from environment variables) and register it at start-up with `set_csam_matcher(...)`. Until you do, `/health` and every response report `csam_hash_matching: NOT_CONFIGURED` and the server logs a warning.
* **Matches must be reported through the legally required channels.** Preserve the content as your provider and legal counsel instruct, do not show it to ordinary moderators, and report it, e.g. to the NCMEC CyberTipline where your provider requires it, and in Tanzania to the Police Force cyber-crime unit and TCRA under the Cybercrimes Act 2015. Get legal advice on your exact obligations.

---

## Tests
```powershell
python -m pytest -v                # everything (about 15 minutes on CPU: loads all models)
python -m pytest tests/test_components.py   # fast unit tests (seconds)
```
* `tests/test_listings.py`: **94 labelled Tanzanian listings** (`tests/data/test_listings.csv`, Swahili/English/mixed): 45 legitimate (including tricky ones such as unga wa sembe, glue gun, Kobe shoes, toy water gun, sukuma wiki photo) and 49 violating (obfuscated slang, contraband photos under innocent titles, contact numbers, photo mismatch …). Legit must be `ALLOW`, BLOCK-labelled must be `BLOCK`.
* `tests/test_components.py` (47 tests): normaliser, lexicon, false-positive protections, contacts, fusion, ESCALATE rule, CSAM interface.
* `tests/test_api_media.py`: `POST /moderate`, multi-file, video frames, unreadable files, hash blocklist, API key, upload limits, `/health`.
* `test_system.py`: the original end-to-end suite (Layer 2, feedback loop, workbench UI).

Tests use a temporary database and a copy of the Layer 2 model (`conftest.py`), so your real `audit_logs.db` and `models/fraud_model.joblib` are not modified. Test photos are public Wikimedia Commons images (`tests/assets/SOURCES.md`).

## Security
* Set **`MODERATION_API_KEY`**: every endpoint except `/health` and the workbench UI then requires the header `X-API-Key`. Without it the server logs a warning and accepts anonymous requests, so keep it bound to `127.0.0.1` or a private network. The Risk Workbench UI does not send a key; use it only for local testing.
* CORS is off by default; add your web origins to `api.cors_origins` only if browsers call the API directly.
* Uploads are limited in count and size, images are capped at 40 MP (decompression-bomb guard), and errors never return internal exception text.
* `/api/v1/feedback` changes the online model, so call it only from your trusted backend.

## Known limitations
* **The accuracy numbers come from listings written for this project, not from your real store traffic.** Real sellers will invent new slang. Label real listings and retrain (see below).
* CLIP is a general model. Leaf-shaped drawings, cartoons or unusual products can land in `REVIEW`, and some disguised photos will be missed. Photo-vs-title mismatch is only checked when the title names a known product type (`translate_swahili_concept_to_english` in `services/moderation_layer1.py`); unknown products report `NOT_VERIFIED`.
* The zero-shot text model is noisy on Swahili. It is therefore only used as a meta-model feature and, at very high confidence, for `REVIEW`, never to block.
* Falconsai/NudeNet can flag swimwear or lingerie photos for `REVIEW`. The age estimator is approximate, which is why it can only *escalate to humans*, never decide alone.
* OCR is English/Latin-script only; stylised fonts on images are often misread.
* Whisper audio transcription is off by default (speed), and `whisper-base` makes mistakes on Swahili; use `whisper-small` or larger for production.
* On CPU a listing takes several seconds; use a GPU for production traffic.
* The Layer 2 River model is still trained on synthetic transactions.

## Project structure
```
├── config/
│   ├── config.json              # thresholds, model names, limits (edit me)
│   ├── lexicon.json             # Swahili/English slang lists (edit me)
│   └── settings.py              # loads both + Layer 2 constants and CLIP taxonomy
├── services/
│   ├── moderation_layer1.py     # content moderation pipeline for one listing
│   ├── text_normalizer.py       # de-obfuscation (b4ng1, b.a.n.g.i, baaangi, homoglyphs)
│   ├── lexicon.py               # lexicon matcher (strong/review/weak/context/allow, typos, emoji)
│   ├── contact_detector.py      # phone numbers, WhatsApp/Telegram, links, Lipa namba
│   ├── text_classifier.py       # zero-shot text model, optional NLLB translation
│   ├── image_analyzer.py        # NSFW, NudeNet, CLIP, OCR, pHash blocklist, age estimate
│   ├── video_analyzer.py        # frame sampling + optional Whisper audio
│   ├── child_safety.py          # ESCALATE rule + CSAM hash-matching interface (PhotoDNA/Safer)
│   ├── fusion.py                # signals -> decision, meta-model, seller messages
│   ├── model_registry.py        # lazy, fail-soft model loading, GPU/CPU
│   ├── moderation_orchestrator.py  # Layer 1 + Layer 2 + audit log
│   └── fraud_model_layer2.py    # River online fraud model
├── api/routes.py, main.py       # FastAPI (POST /moderate, GET /health, /api/v1/...)
├── training/train_meta_model.py # train / evaluate the calibrated meta-model
├── moderate.py                  # CLI
├── download_models.py           # fetch pretrained models
├── data/                        # train_listings.csv, hash_blocklist.json
├── tests/                       # pytest suites, 94-listing test CSV, test photos
├── examples/java/ModerationClient.java
└── static/                      # Risk Workbench UI
```
