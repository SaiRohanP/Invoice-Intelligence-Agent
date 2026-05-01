# app.py
# DAY 5 — Streamlit UI: Upload, Extract, Q&A, and Anomaly Report
# Usage: streamlit run app.py

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# ── Auto-bootstrap pipeline on cold start ────────────────────────────────────
# Streamlit Cloud wipes the filesystem on every restart. This block detects a
# cold start (no extracted_invoices.json) and silently runs the full pipeline
# so invoices are always available without any manual intervention.

# import subprocess

# PROJECT_ROOT = Path(__file__).parent

# def _run(script: str, label: str, progress):
#     result = subprocess.run(
#         ["python", script],
#         cwd=PROJECT_ROOT,
#         capture_output=True,
#         text=True,
#     )
#     if result.returncode != 0:
#         st.error(f"Bootstrap failed at {label}:\n{result.stderr[-800:]}")
#         st.stop()
#     progress.write(f"✅ {label}")

# def _ensure_packages():
#     """Install packages that may be missing from the deployed environment."""
#     required = ["faker", "reportlab", "pymupdf", "fastembed"]
#     for pkg in required:
#         try:
#             __import__(pkg if pkg != "pymupdf" else "fitz") if pkg != "fastembed" else __import__("fastembed")
#         except ImportError:
#             subprocess.check_call(
#                 [sys.executable, "-m", "pip", "install", pkg, "-q"],
#                 cwd=PROJECT_ROOT,
#             )

# def bootstrap_pipeline():
#     """Run the full data pipeline on first start or after a filesystem reset."""
#     extracted = PROJECT_ROOT / "data" / "extracted_invoices.json"
#     chroma    = PROJECT_ROOT / "data" / "chroma_db"

#     if extracted.exists() and chroma.exists():
#         return  # already initialised — nothing to do

#     st.info("⚙️ First-run setup: generating and processing invoices. This takes ~20 seconds...")
#     progress = st.empty()

#     _ensure_packages()
#     progress.write("✅ Dependencies verified")

#     _run("data/generate_documents.py",   "Invoices & POs generated",      progress)
#     _run("extraction/batch_extract.py",  "Invoices extracted (local)",     progress)
#     _run("rag/build_vectorstore.py",     "Vector store built",             progress)
#     _run("anomaly/detector.py",          "Anomaly detection complete",     progress)

#     progress.write("🚀 Setup complete — loading app...")
#     st.rerun()

# bootstrap_pipeline()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Invoice Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f7f4;
        border-left: 4px solid #0d7855;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .flag-high   { border-left: 4px solid #e53e3e; background: #fff5f5;
                   padding: 10px 14px; border-radius: 6px; margin: 6px 0; }
    .flag-medium { border-left: 4px solid #dd6b20; background: #fffaf0;
                   padding: 10px 14px; border-radius: 6px; margin: 6px 0; }
    .flag-low    { border-left: 4px solid #d69e2e; background: #fffff0;
                   padding: 10px 14px; border-radius: 6px; margin: 6px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/invoice.png", width=60)
    st.title("Invoice Intelligence\n Agent")
    st.caption("AI-powered Source-to-Pay automation")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📤 Upload & Extract",
         "💬 Ask Questions",
         "🚨 Anomaly Report",
         "📊 Dashboard"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Claude AI + Groq LLaMa AI + LangChain + ChromaDB")
    st.caption("Observability: LangSmith")


# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_qa_chain():
    """Load vectorstore + QA chain (cached across sessions)."""
    from rag.pipeline import load_vectorstore, build_qa_chain
    vs = load_vectorstore()
    return build_qa_chain(vs)  # returns a callable: fn(question: str) -> dict


def load_json(path: str, default=None):
    if os.path.exists(path):
        return json.load(open(path))
    return default if default is not None else []


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Upload & Extract
# ─────────────────────────────────────────────────────────────────────────────
if page == "📤 Upload & Extract":
    st.header("📤 Upload & Extract Invoice")
    st.caption("Upload an invoice PDF and PyMuPDF extracts all fields automatically.")

    uploaded = st.file_uploader(
        "Drop your invoice PDF here", type=["pdf"], help="Supports standard & GST invoices"
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        with st.spinner("🔍 Extracting data..."):
            try:
                from extraction.extractor import extract_invoice_data
                result = extract_invoice_data(tmp_path)
                os.unlink(tmp_path)
                st.success("✅ Extraction complete!")
            except Exception as e:
                os.unlink(tmp_path)
                st.error(f"Extraction failed: {e}")
                st.stop()

        # Display results
        col1, col2, col3 = st.columns(3)
        col1.metric("Invoice Number",  result.get("invoice_number", "N/A"))
        col2.metric("PO Reference",    result.get("po_reference", "N/A"))
        col3.metric("Invoice Date",    result.get("invoice_date", "N/A"))

        col4, col5, col6 = st.columns(3)
        col4.metric("Vendor",          (result.get("vendor_name") or "N/A")[:25])
        col5.metric("Total Amount",    f"₹{float(result.get('total_amount') or 0):,.2f}")
        col6.metric("Payment Terms",   result.get("payment_terms", "N/A"))

        st.subheader("Tax Breakdown")
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Subtotal", f"₹{float(result.get('subtotal') or 0):,.2f}")
        tc2.metric("CGST",     f"₹{float(result.get('cgst') or 0):,.2f}")
        tc3.metric("SGST",     f"₹{float(result.get('sgst') or 0):,.2f}")

        if result.get("line_items"):
            st.subheader("Line Items")
            import pandas as pd
            df = pd.DataFrame(result["line_items"])
            st.dataframe(df, width='stretch')

        with st.expander("Raw JSON"):
            st.json(result)

    else:
        st.info("Upload an invoice PDF above to get started.")
        st.markdown("""
        **What gets extracted:**
        - Invoice & PO numbers
        - Vendor name & GSTIN
        - Line items with quantities and rates
        - GST breakdown (CGST / SGST / IGST)
        - Payment terms
        """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — RAG Q&A
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 Ask Questions":
    st.header("💬 Ask Questions About Your Invoices")
    st.caption("Natural language queries across all processed invoices using RAG.")

    if not os.path.exists("data/chroma_db"):
        st.warning("⚠️ Vector store not ready yet. Please wait a moment and refresh the page.")
        st.stop()

    # Suggested queries
    st.subheader("Suggested Questions")
    suggestions = [
        "Which vendor has the highest total invoice amount?",
        "How many invoices are above ₹50,000?",
        "List all invoice numbers and their PO references",
        "Which invoices have CGST greater than ₹5,000?",
        "Show me all invoices from this month",
    ]
    row1 = st.columns(3)
    row2 = st.columns(2)
    all_cols = row1 + row2
    for i, suggestion in enumerate(suggestions):
        if all_cols[i].button(suggestion, width='stretch'):
            st.session_state["query"] = suggestion

    st.divider()
    query = st.text_input(
        "Ask anything about your invoices:",
        value=st.session_state.get("query", ""),
        placeholder="e.g. Which vendor appears most frequently?",
    )

    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching invoices..."):
            try:
                qa_chain = load_qa_chain()
                result   = qa_chain(query)
                st.success(result["result"])

                with st.expander("📄 Source invoices used"):
                    for doc in result.get("source_documents", []):
                        st.markdown(f"**{doc.metadata.get('invoice_number', 'N/A')}** — "
                                    f"{doc.metadata.get('vendor_name', 'N/A')} — "
                                    f"₹{doc.metadata.get('total_amount', 0):,.2f}")
            except Exception as e:
                st.error(f"Query failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Anomaly Report
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🚨 Anomaly Report":
    st.header("🚨 Anomaly Detection Report")

    col_run, col_info = st.columns([1, 3])
    if col_run.button("▶ Run Detection Now", type="primary"):
        with st.spinner("Running anomaly checks..."):
            from anomaly.detector import run_detection
            run_detection()
        st.rerun()

    if not os.path.exists("data/anomalies.json"):
        st.info("Click 'Run Detection Now' to analyse your invoices.")
        st.stop()

    report = load_json("data/anomalies.json", {})
    flags  = report.get("flags", [])

    # Summary metrics
    high   = [f for f in flags if f.get("severity") == "HIGH"]
    medium = [f for f in flags if f.get("severity") == "MEDIUM"]
    low    = [f for f in flags if f.get("severity") == "LOW"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Invoices",   report.get("total_invoices", 0))
    m2.metric("🔴 High Severity", len(high))
    m3.metric("🟡 Medium",        len(medium))
    m4.metric("🟠 Low",           len(low))

    st.divider()

    if report.get("ai_summary"):
        st.subheader("AI Audit Summary")
        st.info(report["ai_summary"])

    st.subheader("Flagged Invoices")
    if not flags:
        st.success("✅ No anomalies detected. All invoices passed validation.")
    else:
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_flags   = sorted(flags, key=lambda f: severity_order.get(f.get("severity","LOW"), 2))

        for flag in sorted_flags:
            sev   = flag.get("severity", "LOW")
            color = "flag-high" if sev == "HIGH" else \
                    "flag-medium" if sev == "MEDIUM" else "flag-low"
            icon  = "🔴" if sev == "HIGH" else "🟡" if sev == "MEDIUM" else "🟠"
            st.markdown(
                f'<div class="{color}">'
                f'<strong>{icon} [{flag["type"]}]</strong> — '
                f'<code>{flag["invoice"]}</code><br/>'
                f'{flag["detail"]}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.header("📊 Invoice Batch Dashboard")

    invoices = load_json("data/extracted_invoices.json")
    if not invoices:
        st.warning("No invoice data found yet. Please wait a moment and refresh the page.")
        st.stop()

    import pandas as pd

    df = pd.DataFrame([{
        "Invoice":  i.get("invoice_number", ""),
        "Vendor":   (i.get("vendor_name") or "Unknown")[:30],
        "PO Ref":   i.get("po_reference", ""),
        "Date":     i.get("invoice_date", ""),
        "Subtotal": float(i.get("subtotal") or 0),
        "CGST":     float(i.get("cgst") or 0),
        "SGST":     float(i.get("sgst") or 0),
        "Total":    float(i.get("total_amount") or 0),
    } for i in invoices])

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Invoices",   len(df))
    k2.metric("Total Value",      f"₹{df['Total'].sum():,.0f}")
    k3.metric("Avg Invoice",      f"₹{df['Total'].mean():,.0f}")
    k4.metric("Unique Vendors",   df["Vendor"].nunique())

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top 5 Vendors by Value")
        top_vendors = df.groupby("Vendor")["Total"].sum().nlargest(5).reset_index()
        st.bar_chart(top_vendors.set_index("Vendor"))

    with col_r:
        st.subheader("Invoice Value Distribution")
        bins = pd.cut(df["Total"], bins=[0, 10000, 30000, 60000, 100000, float("inf")],
                      labels=["<10K", "10–30K", "30–60K", "60–100K", ">100K"])
        st.bar_chart(bins.value_counts().sort_index())

    st.divider()
    st.subheader("All Invoices")
    st.dataframe(
        df.style.format({"Subtotal": "₹{:,.2f}", "CGST": "₹{:,.2f}",
                         "SGST": "₹{:,.2f}", "Total": "₹{:,.2f}"}),
        width='stretch',
        height=35 * len(df) + 38,
        column_config={
            "Invoice":  st.column_config.TextColumn("Invoice",  width="small"),
            "Vendor":   st.column_config.TextColumn("Vendor",   width="large"),
            "PO Ref":   st.column_config.TextColumn("PO Ref",   width="small"),
            "Date":     st.column_config.TextColumn("Date",     width="small"),
            "Subtotal": st.column_config.NumberColumn("Subtotal", format="₹%.2f", width="medium"),
            "CGST":     st.column_config.NumberColumn("CGST",     format="₹%.2f", width="small"),
            "SGST":     st.column_config.NumberColumn("SGST",     format="₹%.2f", width="small"),
            "Total":    st.column_config.NumberColumn("Total",    format="₹%.2f", width="medium"),
        },
    )