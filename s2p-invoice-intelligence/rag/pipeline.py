# rag/pipeline.py
# DAY 3 — RAG pipeline: embed invoices into ChromaDB, build Q&A chain
# Usage: python rag/build_vectorstore.py

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate


# ── Free local embeddings — no API cost ─────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR      = "data/chroma_db"


def get_embeddings():
    """Return cached HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def invoice_to_document(inv: dict) -> Document:
    """Convert extracted invoice dict into a LangChain Document."""
    line_items_text = ""
    for item in inv.get("line_items", []):
        line_items_text += (
            f"  • {item.get('description', 'N/A')} | "
            f"Qty: {item.get('quantity', 0)} | "
            f"Rate: {item.get('unit_rate', 0)} | "
            f"Amount: ₹{item.get('amount', 0):,.2f}\n"
        )

    text = f"""
Invoice Number  : {inv.get('invoice_number', 'N/A')}
PO Reference    : {inv.get('po_reference', 'N/A')}
Invoice Date    : {inv.get('invoice_date', 'N/A')}
Vendor Name     : {inv.get('vendor_name', 'N/A')}
Vendor GSTIN    : {inv.get('vendor_gstin', 'N/A')}
Payment Terms   : {inv.get('payment_terms', 'N/A')}

Line Items:
{line_items_text if line_items_text else '  None extracted'}

Subtotal        : ₹{inv.get('subtotal', 0):,.2f}
CGST            : ₹{inv.get('cgst', 0):,.2f}
SGST            : ₹{inv.get('sgst', 0):,.2f}
Total Amount    : ₹{inv.get('total_amount', 0):,.2f}
""".strip()

    return Document(
        page_content=text,
        metadata={
            "invoice_number": str(inv.get("invoice_number", "")),
            "vendor_name":    str(inv.get("vendor_name", "")),
            "po_reference":   str(inv.get("po_reference", "")),
            "total_amount":   float(inv.get("total_amount") or 0),
            "source_file":    str(inv.get("source_file", "")),
        },
    )


def build_vectorstore(extracted_path: str = "data/extracted_invoices.json") -> Chroma:
    """Embed all invoices and persist to ChromaDB."""
    if not os.path.exists(extracted_path):
        raise FileNotFoundError(
            f"{extracted_path} not found. Run batch_extract.py first."
        )

    invoices = json.load(open(extracted_path))
    docs     = [invoice_to_document(inv) for inv in invoices]

    print(f"Embedding {len(docs)} invoices using {EMBEDDING_MODEL}...")
    embeddings  = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="s2p_invoices",
    )
    print(f"✅ ChromaDB built at {CHROMA_DIR} with {len(docs)} documents.")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB from disk."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name="s2p_invoices",
    )


S2P_QA_PROMPT = PromptTemplate.from_template("""
You are an expert Accounts Payable analyst with deep knowledge of S2P workflows.
Use the retrieved invoice data below to answer the question accurately and concisely.
If the data does not contain enough information, say so clearly.

Retrieved Invoice Data:
{context}

Question: {question}

Answer (be specific, mention invoice numbers and amounts where relevant):
""")


def build_qa_chain(vectorstore: Chroma) -> RetrievalQA:
    """Build a RetrievalQA chain over the invoice vectorstore."""
    llm       = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": S2P_QA_PROMPT},
    )


if __name__ == "__main__":
    vs = build_vectorstore()
    qa = build_qa_chain(vs)

    test_questions = [
        "Which vendor has the highest total invoice amount?",
        "How many invoices are above ₹50,000?",
        "List all invoice numbers and their PO references.",
        "Which invoices have CGST greater than ₹5,000?",
    ]
    print("\n── Testing Q&A chain ───────────────────────────────\n")
    for q in test_questions:
        print(f"Q: {q}")
        result = qa({"query": q})
        print(f"A: {result['result']}\n")

    print("Next step: python anomaly/detector.py")
