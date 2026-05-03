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
