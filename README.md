# pamostore fraud check — Fraud Detection & Content Moderation Engine

A self-learning fraud detection and illegal-goods moderation service for **Tanzanian social commerce and escrow** (TZS, mobile money, rider delivery). It checks every listing and payment and returns a clear decision: **`ALLOW`**, **`HOLD_FOR_REVIEW`**, or **`BLOCK`**.

It ships with a browser-based **Risk Workbench** where you can try scenarios in one click. No coding needed.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688) ![River](https://img.shields.io/badge/River-online%20ML-orange)

---

## Contents

- [What it does](#what-it-does)
- [Quick start (5 minutes)](#quick-start-5-minutes)
- [Using the Risk Workbench](#using-the-risk-workbench)
- [API usage](#api-usage)
- [Configuration](#configuration)
- [How decisions are made](#how-decisions-are-made)
- [Running the tests](#running-the-tests)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations & production checklist](#limitations--production-checklist)

---

## What it does

Every request passes through two layers:

| Layer | What it checks | How |
|---|---|---|
| **1. Content moderation** | Prohibited items and coded slang in **Swahili & English** (e.g. `bangi`, `cha arusha`, `bastola`, `noti bandia`, `upatu`)<br>Context-aware: `unga wa sembe`, `kisu cha jikoni`, `majani ya chai` are *not* flagged<br>Photo vs. title mismatch (e.g. "Gauni" listed with a photo of a men's suit or a screenshot)<br>Drugs, weapons, and NSFW content in images | Keyword lexicon + context rules<br>Zero-shot vision model ([CLIP](https://huggingface.co/openai/clip-vit-base-patch32)) over 36 retail and safety categories, with a Swahili→English product translator |
| **2. Transaction fraud** | Price vs. category market median, time of day, account age, order history, dispute rate, repeat-buyer velocity | Online machine learning ([River](https://riverml.xyz)) that **learns from every piece of feedback** without retraining from scratch |

All decisions are written to a local SQLite audit log.

---

## Quick start (5 minutes)

### Requirements

- **Python 3.10 or newer**. [Download](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** during install.
- **About 3 GB of free disk space** (PyTorch plus the vision model).
- An internet connection for the first run.

### 1. Get the code

```bash
git clone https://github.com/Denyjoe/fraud-Detection-Engine.git
cd fraud-Detection-Engine
```

No Git? Click **Code → Download ZIP** on GitHub, unzip it, and open a terminal in that folder.

### 2. Run the setup script (one command)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

**macOS / Linux:**
```bash
bash setup.sh
```

The script checks your Python version, creates an isolated `.venv`, installs dependencies, and pre-trains the fraud model on 2,000 synthetic Tanzanian transactions. It is safe to run again.

### 3. Start the server

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

**macOS / Linux:**
```bash
bash run.sh
```

### 4. Open it

| What | URL |
|---|---|
| Risk Workbench (UI) | http://127.0.0.1:8000 |
| Interactive API docs | http://127.0.0.1:8000/docs |
| Health check | http://127.0.0.1:8000/health |

> **First start is slower.** The vision model (~600 MB) downloads once and is cached. Later starts take a few seconds.

<details>
<summary><b>Manual installation (if you prefer not to use the scripts)</b></summary>

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_synthetic_data.py      # pre-train the fraud model
uvicorn main:app --host 127.0.0.1 --port 8000
```

On Linux, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` before installing the requirements to get the smaller CPU-only PyTorch (~2 GB saved).
</details>

---

## Using the Risk Workbench

1. Click a **preset** at the top: *Sinza Boutique*, *Mwenge Type-C*, *Unga Sembe 5kg*, *Disguised Contraband 3 AM*, or *Borderline Checkout*. The form fills in and is evaluated automatically.
2. Or fill the form yourself: title, description (Swahili or English), price in TZS, time of day, and seller history. You can also **drag and drop a product photo**.
3. The right panel shows:
   - the **verdict** and fraud probability, with the reasons behind it
   - Swahili slang risk and matched terms
   - **Seller claimed vs. AI detected in image**, with a `MATCH` / `MISMATCH` / `CONTRABAND` badge
   - the category price benchmark and the ML probability
4. Teach the model: click **Confirm Legit (0)** or **Flag Fraud (1)** and watch the probability shift live (online learning).
5. The **Audit Ledger** at the bottom shows the latest decisions from the database.

---

## API usage

Base URL: `http://127.0.0.1:8000/api/v1`. Full interactive docs are at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/moderate-listing` | Check a new listing (JSON, optional base64 image) |
| `POST` | `/moderate-listing-upload` | Same as above, with multipart file upload |
| `POST` | `/predict-transaction` | Score an escrow payment before funds are locked |
| `POST` | `/feedback` | Send the real outcome (`0` = legit, `1` = fraud); the model learns instantly |
| `GET` | `/metrics` | Totals and running model accuracy / F1 |
| `GET` | `/audit-stats` | Latest 15 audit entries plus metrics |

### Check a listing

```bash
curl -X POST http://127.0.0.1:8000/api/v1/moderate-listing \
  -H "Content-Type: application/json" \
  -d '{
    "listing_id": "LIST-KTG-01",
    "seller_id": "USR-KARIAKOO-99",
    "title": "Gauni Nzuri ya Vitenge Kariakoo",
    "description": "Nguo bora ya pamba ya asili kutoka Kariakoo, rangi ya bluu.",
    "price_tzs": 45000,
    "account_age_days": 180,
    "seller_completed_orders": 75,
    "seller_dispute_rate": 0.01,
    "hour_of_day": 14,
    "is_repeat_buyer_same_day": 0
  }'
```

Add `"image_base64": "<base64 string>"` to include a photo. `category` is optional (`fashion`, `electronics`, `groceries`, `home_goods`, `beauty`, `general`) and is auto-detected if omitted.

Response (abridged):
```json
{
  "listing_id": "LIST-KTG-01",
  "category": "fashion",
  "decision": "ALLOW",
  "fraud_probability": 0.0321,
  "risk_reasons": [],
  "seller_claimed_concept": "a women's dress, gown, or African print kitenge dress",
  "visual_comparison_status": "NO_IMAGE",
  "layer1_details": { "swahili_slang_risk_score": 0.0, "image_text_mismatch_score": 0.0, "...": "..." },
  "layer2_features": { "price_to_category_median_ratio": 1.0, "...": "..." },
  "audit_log_id": 1
}
```

### Score a payment

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict-transaction \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX-TZS-88912",
    "seller_id": "USR-SINZA-02",
    "buyer_id": "USR-BUYER-55",
    "price_tzs": 120000,
    "category": "electronics",
    "hour_of_day": 13,
    "account_age_days": 45,
    "seller_completed_orders": 12,
    "seller_dispute_rate": 0.03,
    "is_repeat_buyer_same_day": 0
  }'
```

Returns `decision`, `fraud_probability`, and the `features` used. Pass those `features` back to `/feedback`.

### Teach the model

```bash
curl -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX-TZS-88912",
    "actual_label": 1,
    "features": { "category": "electronics", "price_tzs": 120000, "price_to_category_median_ratio": 0.3429,
                  "hour_of_day": 13, "account_age_days": 45, "seller_completed_orders": 12,
                  "seller_dispute_rate": 0.03, "image_text_mismatch_score": 0.1,
                  "swahili_slang_risk_score": 0.0, "is_repeat_buyer_same_day": 0 },
    "comment": "Rider reported empty box"
  }'
```

Response: `{ "previous_probability": 0.1194, "updated_probability": 0.1238, "delta": 0.0044, ... }`

---

## Configuration

All settings have sensible defaults. Override them with environment variables:

| Variable | Default | Description |
|---|---|---|
| `FRAUD_MODEL_PATH` | `models/fraud_model.joblib` | Where the River model is stored |
| `AUDIT_DB_PATH` | `audit_logs.db` | SQLite audit log location |
| `CLIP_MODEL_NAME` | `openai/clip-vit-base-patch32` | Hugging Face vision model |
| `USE_REAL_CLIP` | `true` | Set to `false` to run without the vision model (text and ML checks still work; image checks report "unavailable") |
| `HF_HOME` | your user cache folder | Where downloaded models are cached. Point it at a drive with more space if needed. |

Example (Windows PowerShell): `$env:USE_REAL_CLIP="false"; .\run.ps1`
Example (macOS/Linux): `USE_REAL_CLIP=false bash run.sh`

Decision thresholds, category price medians, and trust rules live in [`config/settings.py`](config/settings.py) as named constants.

---

## How decisions are made

| Fraud probability | Decision |
|---|---|
| below 40% | `ALLOW` |
| 40% – 75% | `HOLD_FOR_REVIEW` |
| above 75% | `BLOCK` |

Guardrails on top of the ML score keep honest sellers safe and contraband out:

- **Hard block:** confirmed contraband or NSFW (from text or image) is always `BLOCK`.
- **Corroborated block:** ML score > 85% is only blocked when backed by a high dispute rate (> 25%) or a severe photo mismatch (> 0.80). Otherwise it goes to review.
- **No unfair blocks:** an expensive item listed late at night by a clean seller with a matching photo is never auto-blocked.
- **Trusted merchants** (≥ 60 days old, ≥ 10 orders, ≤ 5% disputes) get a 30% risk reduction.
- **Photo mismatch or screenshot** uploads can never be auto-allowed.
- **Price is judged per category:** a 2,500,000 TZS iPhone is normal; a 2,500,000 TZS dress is not.

---

## Running the tests

```bash
# Windows
.venv\Scripts\python -m pytest -v test_system.py
# macOS / Linux
.venv/bin/python -m pytest -v test_system.py
```

The 21 end-to-end tests cover: legit Kariakoo/Sinza sales, coded contraband disguised as a dress, a cannabis image under an innocent title, a men's suit photo for a "Gauni" listing, screenshot uploads, `unga wa sembe` and Type-C chargers not being falsely flagged, the online-learning feedback loop, input validation, and the UI and audit endpoints.

> Tests run against your local `audit_logs.db` and `models/fraud_model.joblib` and add entries to them. Re-run `setup` with `-Retrain` / `--retrain` to reset the model.

---

## Project structure

```
├── api/routes.py                  # REST endpoints
├── config/settings.py             # All thresholds, taxonomy & env settings
├── database/                      # SQLite connection + repository (audit log)
├── schemas/api_schemas.py         # Validated request/response models
├── services/
│   ├── moderation_layer1.py       # Swahili slang + vision (CLIP) moderation
│   ├── fraud_model_layer2.py      # River online fraud model
│   └── moderation_orchestrator.py # Combines layers + guardrails
├── static/                        # Risk Workbench UI (index.html, app.js)
├── models/                        # Trained model is created here by setup
├── generate_synthetic_data.py     # Builds 2,000 synthetic records & pre-trains
├── main.py                        # FastAPI app entrypoint
├── test_system.py                 # End-to-end test suite
├── setup.ps1 / setup.sh           # One-command setup
└── run.ps1 / run.sh               # One-command start
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `running scripts is disabled on this system` | Use the full command shown above: `powershell -ExecutionPolicy Bypass -File .\setup.ps1` |
| `Python 3.10+ was not found` | Install Python from python.org and tick **Add to PATH**, then open a **new** terminal |
| `not enough space on the disk` | Free ~3 GB, or set `HF_HOME` to a drive with space, or run with `USE_REAL_CLIP=false` |
| First start seems stuck | It is downloading the vision model (~600 MB). Wait for "CLIP model successfully loaded" in the log. |
| `address already in use` / port 8000 busy | Use another port: `.\run.ps1 -Port 8080` or `PORT=8080 bash run.sh` |
| Model predictions look untrained | Re-run setup with `-Retrain` (Windows) or `--retrain` (macOS/Linux) |

---

## Limitations & production checklist

This is a strong prototype, **not yet a hardened production service**. Before going live:

- [ ] **Add authentication** (API keys or OAuth) to every endpoint. There is none today, and CORS allows all origins. The run scripts bind to `127.0.0.1` by default for this reason.
- [ ] **Train on real data.** The cold-start model learns from *synthetic* records, so its ~99.9% training accuracy reflects that synthetic data, not real-world performance.
- [ ] **Validate the vision model on real product photos.** CLIP zero-shot accuracy on fine-grained clothing and on screenshots is limited. A SigLIP upgrade is planned.
- [ ] Move from SQLite to a managed database for multi-instance deployments.
- [ ] Add rate limiting and request size limits for image uploads.
