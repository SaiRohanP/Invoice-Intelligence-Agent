# 📄 S2P Invoice Intelligence

An AI-powered Source-to-Pay invoice automation system featuring **multimodal document extraction**, **RAG-based Q&A**, and **rule + LLM-powered anomaly detection**. Built to mirror real-world S2P automation workflows in enterprise procurement and accounts payable.

---

## 🏗️ Architecture

```
User Interface (Streamlit)
        │
        ├── Upload & Extract   →  Claude Vision (multimodal PDF → JSON)
        ├── Ask Questions      →  LangChain RAG + ChromaDB (natural language Q&A)
        ├── Anomaly Report     →  Rule-based + Claude LLM audit
        └── Dashboard          →  Batch analytics + vendor insights
                                        │
                               LangSmith (observability + tracing)
```

## ✨ Features

| Feature | Technology | IBM JD Keyword |
|---|---|---|
| Multimodal invoice extraction | Claude Vision API | AI/ML, agentic workflows |
| RAG Q&A over invoice batch | LangChain + ChromaDB | Pipelines, ETL |
| 3-type anomaly detection | Rule engine + Claude | Workflow automation |
| AI audit summary | Claude Haiku | LLMOps |
| Observability tracing | LangSmith | Monitoring, observability |
| Synthetic data generation | Faker + ReportLab | Data validation |

## 🗂️ Project Structure

```
s2p-invoice-intelligence/
├── data/
│   └── generate_documents.py    # Day 1 — synthetic invoice + PO generation
├── extraction/
│   ├── extractor.py             # Day 2 — Claude Vision extraction
│   └── batch_extract.py         # Day 2 — batch processing
├── rag/
│   ├── pipeline.py              # Day 3 — ChromaDB + LangChain RAG
│   └── build_vectorstore.py     # Day 3 — build embeddings
├── anomaly/
│   └── detector.py              # Day 4 — rule checks + AI audit
├── app.py                       # Day 5 — Streamlit UI
├── requirements.txt
└── .env.example
```

## 🚀 Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/s2p-invoice-intelligence.git
cd s2p-invoice-intelligence
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# ANTHROPIC_API_KEY  →  https://console.anthropic.com  (free credits on signup)
# LANGCHAIN_API_KEY  →  https://smith.langchain.com    (free tier)
```

### 3. Run the pipeline day by day

```bash
# Day 1 — Generate synthetic invoices and POs
python data/generate_documents.py

# Day 2 — Extract all invoices using Claude Vision
python extraction/batch_extract.py

# Day 3 — Build the RAG vector store
python rag/build_vectorstore.py

# Day 4 — Run anomaly detection
python anomaly/detector.py

# Day 5 — Launch the Streamlit UI
streamlit run app.py
```

## 💡 Anomaly Types Detected

| Type | Description | Severity |
|---|---|---|
| `DUPLICATE_INVOICE` | Same invoice number appears twice | HIGH |
| `AMOUNT_MISMATCH` | Invoice total exceeds PO approved amount by >5% | HIGH |
| `INVALID_PO_REFERENCE` | PO reference not found in PO master | MEDIUM |
| `MISSING_FIELDS` | Required fields (vendor, GSTIN, total) are empty | MEDIUM |
| `INVALID_GSTIN` | GSTIN format does not match 15-char standard | LOW |

## 🔍 Example Q&A Queries

```
"Which vendor has the highest total invoice amount?"
"How many invoices are above ₹50,000?"
"List all invoices with their PO references"
"Which invoices have CGST greater than ₹5,000?"
"Show invoices where payment terms are Net 60"
```

## 🌐 Deploy for Free

1. Push to a **public** GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets in Streamlit Cloud settings:
   - `ANTHROPIC_API_KEY`
   - `LANGCHAIN_API_KEY`
   - `LANGCHAIN_TRACING_V2 = true`
5. Deploy — you get a public URL instantly

## 🛠️ Tech Stack

- **Claude API** (Anthropic) — Vision extraction + LLM audit
- **LangChain** — RAG pipeline, prompt management
- **ChromaDB** — Local vector store for invoice embeddings
- **HuggingFace Sentence Transformers** — Free local embeddings
- **LangSmith** — LLM observability and tracing
- **Streamlit** — Web UI + free deployment
- **Faker + ReportLab** — Synthetic document generation

## 📈 Observability

All LLM calls are automatically traced in [LangSmith](https://smith.langchain.com). You can monitor:
- Token usage and API costs per query
- Retrieval latency and chunk quality
- Agent reasoning traces
- End-to-end pipeline performance

## 🔮 Planned Extensions

- [ ] 3-way PO + GRN + Invoice matching
- [ ] FastAPI microservice layer
- [ ] Event-driven webhook triggers
- [ ] Docker containerisation + GitHub Actions CI/CD
- [ ] PostgreSQL persistent store
- [ ] IBM Watsonx Orchestrate integration

---

*Built as a portfolio project to demonstrate S2P automation, agentic AI workflows, and enterprise integration patterns.*
