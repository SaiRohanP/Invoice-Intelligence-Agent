# 📄 S2P Invoice Intelligence — Complete Installation & Deployment Guide

> **Stack:** Python 3.10+ · Streamlit · Claude Vision API · LangChain · ChromaDB · HuggingFace Embeddings · LangSmith  
> **Deployment targets covered:** Local dev → Streamlit Cloud (free) → Docker + VPS (production)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites & API Keys](#1-prerequisites--api-keys)
3. [Local Development Setup](#2-local-development-setup)
4. [Running the Pipeline (Step-by-Step)](#3-running-the-pipeline-step-by-step)
5. [Launch the Streamlit UI](#4-launch-the-streamlit-ui)
6. [Deploy to Streamlit Cloud (Free, Recommended)](#5-deploy-to-streamlit-cloud-free-recommended)
7. [Docker Deployment (Self-Hosted / VPS)](#6-docker-deployment-self-hosted--vps)
8. [Environment Variables Reference](#7-environment-variables-reference)
9. [Troubleshooting](#8-troubleshooting)
10. [Post-Deployment Checklist](#9-post-deployment-checklist)

---

## Architecture Overview

```
User Browser
     │
     ▼
Streamlit UI  (app.py)
     │
     ├── Upload & Extract   ──► Claude Vision API (claude-opus-4-5)
     │                            PDF → PNG via PyMuPDF → base64 → JSON
     │
     ├── Ask Questions      ──► LangChain RetrievalQA
     │                            HuggingFace all-MiniLM-L6-v2 (local embeddings)
     │                            ChromaDB (persisted at data/chroma_db/)
     │                            Claude Haiku (answer generation)
     │
     ├── Anomaly Report     ──► Rule engine + Claude Haiku audit summary
     │                            Reads extracted_invoices.json + pos_meta.json
     │                            Writes anomalies.json
     │
     └── Dashboard          ──► Pandas + Streamlit charts from extracted_invoices.json
                                         │
                                LangSmith tracing (all LangChain calls auto-traced)
```

**Data flow at a glance:**

```
generate_documents.py
        │
        ▼
data/invoices/*.pdf  +  data/pos_meta.json
        │
        ▼
batch_extract.py  ──►  data/extracted_invoices.json
        │
        ▼
build_vectorstore.py  ──►  data/chroma_db/
        │
        ▼
detector.py  ──►  data/anomalies.json
        │
        ▼
streamlit run app.py  ──►  http://localhost:8501
```

---

## 1. Prerequisites & API Keys

### System requirements

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended; 3.12 works |
| RAM | 4 GB | 8 GB recommended (HuggingFace model download ~90 MB) |
| Disk | 1 GB free | Synthetic PDFs + ChromaDB + model cache |
| Internet | Required | For API calls and first-time model download |

### Required API keys

**A. Anthropic API Key** (used for Claude Vision extraction and Haiku audit)

1. Go to [https://console.anthropic.com](https://console.anthropic.com)
2. Sign up / log in → **API Keys** → **Create Key**
3. Copy the key — you will only see it once
4. Free credits are available on new accounts; the project uses:
   - `claude-opus-4-5` for invoice extraction (Vision)
   - `claude-haiku-4-5-20251001` for RAG answers + audit summary (cheap)

**B. LangSmith API Key** (optional but highly recommended for observability)

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up with GitHub / Google (free tier is generous)
3. **Settings → API Keys → Create API Key**
4. Copy the key

> 💡 If you skip LangSmith, set `LANGCHAIN_TRACING_V2=false` in your `.env` — the app still works fully.

---

## 2. Local Development Setup

### Step 1 — Get the source code

If you have the zip file:
```bash
unzip s2p-invoice-intelligence.zip
cd s2p-invoice-intelligence
```

Or clone from GitHub (after you push it):
```bash
git clone https://github.com/YOUR_USERNAME/s2p-invoice-intelligence.git
cd s2p-invoice-intelligence
```

### Step 2 — Create a Python virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Windows (Command Prompt)
venv\Scripts\activate.bat
```

You should see `(venv)` prepended to your prompt.

### Step 3 — Install all dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ **First install takes 3–5 minutes.** The largest downloads are:
> - `sentence-transformers` (~400 MB including PyTorch)
> - `chromadb` with its native binaries
> - `pymupdf` (PyMuPDF) for PDF-to-image conversion

If you hit a build error on `chromadb` or `sentence-transformers` on Windows, install the C++ Build Tools:
```
winget install Microsoft.VisualStudio.2022.BuildTools
```

### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in your keys:

```dotenv
# .env
ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

LANGCHAIN_API_KEY=ls__XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=s2p-invoice-intelligence
```

**Verify the setup:**
```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
print('Anthropic key set:', bool(os.getenv('ANTHROPIC_API_KEY')))
print('LangSmith key set:', bool(os.getenv('LANGCHAIN_API_KEY')))
"
```

Both should print `True`.

---

## 3. Running the Pipeline (Step-by-Step)

Run these four scripts **in order** before launching the UI. Each script produces output files that the next step depends on.

### Step A — Generate synthetic invoices and POs

```bash
python data/generate_documents.py
```

**What it does:**
- Creates 8 synthetic vendor profiles with fake Indian GSTINs
- Generates ~15 invoice PDFs at `data/invoices/INV-0001.pdf` … `INV-0015.pdf`
- Generates matching PO PDFs at `data/pos/PO-0001.pdf` … `PO-0015.pdf`
- Writes `data/pos_meta.json` (PO master with approved amounts — needed by anomaly detector)

**Expected output:**
```
Generating 15 invoice + PO pairs...
✅  data/invoices/INV-0001.pdf
✅  data/invoices/INV-0002.pdf
...
✅ Saved PO metadata → data/pos_meta.json
Done! 15 invoices and 15 POs generated.
```

> 📁 After this step you should have `data/invoices/` with 15 PDFs and `data/pos_meta.json`.

---

### Step B — Extract invoices with Claude Vision

```bash
python extraction/batch_extract.py
```

**What it does:**
- Converts each invoice PDF's first page to a base64 PNG using PyMuPDF
- Sends each image to `claude-opus-4-5` with a structured extraction prompt
- Parses the JSON response and saves all results to `data/extracted_invoices.json`
- A 0.5 s delay between calls respects Anthropic rate limits

**Expected output:**
```
Found 15 invoices to extract.

[01/15] Extracting INV-0001.pdf...  ✓  Invoice: INV-0001 | Vendor: Tech Corp Ltd | Total: ₹45,200
[02/15] Extracting INV-0002.pdf...  ✓  Invoice: INV-0002 | Vendor: Supply Chain Co | Total: ₹82,400
...
✅ Extracted 15 invoices → data/extracted_invoices.json
```

> ⏱️ This step makes 15 Claude Vision API calls — takes ~1–2 minutes.  
> 💰 Each call uses ~1,000 tokens. Total cost ≈ $0.01–$0.05 for 15 invoices.

**If a PDF fails to extract**, check `data/extraction_failures.json` for the error detail.

---

### Step C — Build the RAG vector store

```bash
python rag/build_vectorstore.py
```

**What it does:**
- Converts each extracted invoice dict into a structured text `Document`
- Embeds all 15 documents using `sentence-transformers/all-MiniLM-L6-v2` (runs **locally**, no API cost)
- Persists the ChromaDB collection to `data/chroma_db/`

**Expected output:**
```
Embedding 15 invoices using sentence-transformers/all-MiniLM-L6-v2...
✅ ChromaDB built at data/chroma_db with 15 documents.

Vectorstore ready. Now run: python anomaly/detector.py
```

> 📥 First run downloads the embedding model (~90 MB) to `~/.cache/huggingface/`. Subsequent runs use the cache.

---

### Step D — Run anomaly detection

```bash
python anomaly/detector.py
```

**What it does:**
- Runs 4 rule-based checks:  
  `DUPLICATE_INVOICE` | `MISSING_FIELDS` | `AMOUNT_MISMATCH` | `INVALID_GSTIN`
- Calls Claude Haiku to generate an AI audit summary of all flagged anomalies
- Saves the full report to `data/anomalies.json`

**Expected output:**
```
Running anomaly checks on 15 invoices...

⚠️  Found 3 anomalies:

  [AMOUNT_MISMATCH] × 1
    → INV-0007: Invoice total ₹95,000 exceeds PO PO-0007 approved amount ₹88,000 by 8.0%.
  [INVALID_GSTIN] × 2
    → INV-0003: GSTIN '27AABC12345Z1Z6' is not 15 characters.
    → INV-0011: GSTIN '29XYZ98' is not 15 characters.

Generating AI audit summary...

── AI Audit Summary ──────────────────────────────────
[AI-generated executive summary appears here]

✅ Anomaly report saved → data/anomalies.json
Next step: streamlit run app.py
```

---

## 4. Launch the Streamlit UI

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser automatically.

### What each page does

| Page | What to expect |
|---|---|
| **📤 Upload & Extract** | Drag-drop any invoice PDF; Claude Vision extracts and displays all fields live |
| **💬 Ask Questions** | Natural language Q&A over your 15 extracted invoices via RAG |
| **🚨 Anomaly Report** | Click "Run Detection Now" to run the full rule + AI audit; view flagged invoices |
| **📊 Dashboard** | KPIs, top-vendor chart, value distribution histogram over the full batch |

---

## 5. Deploy to Streamlit Cloud (Free, Recommended)

This is the fastest path to a **public shareable URL** with zero server management.

### Step 1 — Push to GitHub

```bash
# Initialise git (if not already)
git init
git add .
git commit -m "Initial commit: S2P Invoice Intelligence"

# Create a new repo on github.com (do NOT initialise with README)
git remote add origin https://github.com/YOUR_USERNAME/s2p-invoice-intelligence.git
git branch -M main
git push -u origin main
```

> ⚠️ **Critical:** Make sure `.gitignore` excludes `.env`, `data/chroma_db/`, `data/invoices/`, and `data/extracted_invoices.json` — never commit API keys or generated data.

Your `.gitignore` should contain at minimum:
```gitignore
.env
venv/
__pycache__/
*.pyc
data/invoices/
data/pos/
data/chroma_db/
data/extracted_invoices.json
data/anomalies.json
data/pos_meta.json
.streamlit/secrets.toml
```

### Step 2 — Connect to Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repo → branch `main` → main file: `app.py`
4. Click **"Advanced settings"**

### Step 3 — Add secrets

In the "Advanced settings" → **Secrets** panel, paste:

```toml
ANTHROPIC_API_KEY = "sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
LANGCHAIN_API_KEY = "ls__XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "s2p-invoice-intelligence"
```

### Step 4 — Handle data bootstrap on Streamlit Cloud

Streamlit Cloud has an **ephemeral filesystem** — generated data files do not persist across restarts. Add this bootstrap block at the top of `app.py` just after the imports:

```python
# app.py — add after load_dotenv()
import subprocess, pathlib

def bootstrap_data():
    """Auto-generate data on first run in cloud environments."""
    if not pathlib.Path("data/extracted_invoices.json").exists():
        with st.spinner("🔧 First-run setup: generating sample data..."):
            subprocess.run(["python", "data/generate_documents.py"], check=True)
            subprocess.run(["python", "extraction/batch_extract.py"], check=True)
            subprocess.run(["python", "rag/build_vectorstore.py"], check=True)
            subprocess.run(["python", "anomaly/detector.py"], check=True)

bootstrap_data()
```

> ⚠️ This auto-bootstrap makes ~15 Claude API calls on first deploy (~$0.05). Wrap it behind a `st.button` if you want manual control.

### Step 5 — Deploy

Click **"Deploy!"** — Streamlit installs your `requirements.txt` and starts the app. You get a URL like `https://YOUR-APP.streamlit.app`.

---

## 6. Docker Deployment (Self-Hosted / VPS)

For persistent storage and a stable production environment on any cloud VPS (DigitalOcean, AWS EC2, Hetzner, etc.).

### Step 1 — Create a Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

# System deps for PyMuPDF and chromadb
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the HuggingFace embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
```

### Step 2 — Create docker-compose.yml

```yaml
version: "3.9"
services:
  s2p-app:
    build: .
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./data:/app/data       # Persist generated files across restarts
    restart: unless-stopped
```

### Step 3 — Build and run

```bash
# Build the image (first time: 5–10 min, downloads all deps)
docker compose build

# Start the container
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

App is live at `http://YOUR_SERVER_IP:8501`.

### Step 4 — Run the pipeline inside the container

```bash
# Generate data
docker compose exec s2p-app python data/generate_documents.py

# Extract invoices
docker compose exec s2p-app python extraction/batch_extract.py

# Build vector store
docker compose exec s2p-app python rag/build_vectorstore.py

# Run anomaly detection
docker compose exec s2p-app python anomaly/detector.py
```

### Optional: Nginx reverse proxy with HTTPS

```nginx
# /etc/nginx/sites-available/s2p-app
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass         http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;
    }
}
```

Get a free SSL cert: `sudo certbot --nginx -d yourdomain.com`

---

## 7. Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Yes | — | Claude Vision + Haiku API access |
| `LANGCHAIN_API_KEY` | ⚠️ Optional | — | LangSmith observability tracing |
| `LANGCHAIN_TRACING_V2` | ⚠️ Optional | `false` | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_PROJECT` | ⚠️ Optional | `default` | LangSmith project name for grouping traces |

**Models used in code:**

| File | Model | Purpose | Approx cost |
|---|---|---|---|
| `extraction/extractor.py` | `claude-opus-4-5` | Vision extraction from PDF | ~$0.003/invoice |
| `rag/pipeline.py` | `claude-haiku-4-5-20251001` | RAG answer generation | ~$0.0002/query |
| `anomaly/detector.py` | `claude-haiku-4-5-20251001` | AI audit summary | ~$0.001/run |

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'fitz'`
PyMuPDF installs as `pymupdf` but imports as `fitz`:
```bash
pip uninstall pymupdf fitz
pip install pymupdf
```

### `chromadb` build fails on Windows
Install Visual C++ Build Tools, then:
```bash
pip install chromadb --no-build-isolation
```

### `sentence-transformers` download hangs
The first run downloads ~90 MB. If it times out, set a custom cache:
```bash
export TRANSFORMERS_CACHE=/tmp/hf_cache
```

### Claude API returns `529 Overloaded`
The batch extractor already has a 0.5 s delay. Increase it in `batch_extract.py`:
```python
time.sleep(1.5)   # change from 0.5
```

### Streamlit Cloud: `data/chroma_db does not exist`
The filesystem is reset on every Streamlit Cloud restart. Use the bootstrap block in [Step 4 of Section 5](#step-4--handle-data-bootstrap-on-streamlit-cloud) to auto-regenerate data.

### `LANGCHAIN_API_KEY` warning even when tracing is off
Add to your `.env`:
```
LANGCHAIN_TRACING_V2=false
```

### `ValueError: Input should be a valid dictionary` in LangChain RetrievalQA
This is a LangChain v0.2 breaking change. Ensure you call the chain with:
```python
qa_chain.invoke({"query": question})   # not qa_chain({"query": question})
```

---

## 9. Post-Deployment Checklist

```
Local development
  ☐ venv activated and all packages installed
  ☐ .env file populated with both API keys
  ☐ All 4 pipeline scripts run in order without errors
  ☐ Streamlit app opens at localhost:8501
  ☐ Invoice upload + extraction works on a real PDF
  ☐ RAG Q&A returns answers
  ☐ Anomaly detection flags and AI summary visible
  ☐ Dashboard shows bar charts and invoice table

Streamlit Cloud deployment
  ☐ Repo pushed to GitHub (API keys NOT committed)
  ☐ .gitignore covers .env, data/, venv/, chroma_db/
  ☐ Both API keys added to Streamlit Cloud secrets panel
  ☐ Bootstrap block added to app.py for first-run setup
  ☐ Deployed URL accessible and app loads without errors
  ☐ LangSmith dashboard shows traces for test queries

Docker / VPS deployment
  ☐ Dockerfile and docker-compose.yml created
  ☐ .env file present on server (not in repo)
  ☐ docker compose build succeeds
  ☐ docker compose up -d starts the container
  ☐ Pipeline scripts executed inside container
  ☐ Nginx reverse proxy configured (optional)
  ☐ HTTPS cert obtained via Let's Encrypt (optional)
  ☐ Health check endpoint responding at /_stcore/health
```

---

*Guide covers code as of April 2026. For updates to the planned extensions (FastAPI layer, Docker CI/CD, PostgreSQL, Watsonx integration), refer to the project README.*
