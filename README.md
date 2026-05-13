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

| Feature | Technology |
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
│   └── generate_documents.py    
├── extraction/
│   ├── extractor.py             
│   └── batch_extract.py         
├── rag/
│   ├── pipeline.py              
│   └── build_vectorstore.py     
├── anomaly/
│   └── detector.py              
├── app.py                       
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

---

DEMO VIDEO: https://www.linkedin.com/feed/update/urn:li:activity:7456000946289004544/

*Built as a portfolio project to demonstrate S2P automation, agentic AI workflows, and enterprise integration patterns.*
