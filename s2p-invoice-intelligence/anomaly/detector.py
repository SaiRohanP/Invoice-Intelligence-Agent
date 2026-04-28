# anomaly/detector.py
# DAY 4 — Rule-based + LLM-powered anomaly detection with LangSmith tracing
# Usage: python anomaly/detector.py

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

# LangSmith auto-traces all LangChain calls when these env vars are set.
# No extra code needed — just set LANGCHAIN_API_KEY + LANGCHAIN_TRACING_V2=true

OVERCHARGE_THRESHOLD = 0.05   # flag if invoice > PO by more than 5%
REQUIRED_FIELDS      = [
    "invoice_number", "po_reference", "vendor_name",
    "vendor_gstin", "total_amount",
]


# ── Rule-based checks ────────────────────────────────────────────────────────

def check_duplicates(invoices: list) -> list:
    """Flag invoices with the same invoice_number."""
    seen   = {}
    flags  = []
    for inv in invoices:
        inv_id = inv.get("invoice_number", "")
        if inv_id in seen:
            flags.append({
                "invoice":  inv_id,
                "type":     "DUPLICATE_INVOICE",
                "severity": "HIGH",
                "detail":   f"Invoice {inv_id} appears more than once in the batch.",
                "file":     inv.get("source_file", ""),
            })
        else:
            seen[inv_id] = True
    return flags


def check_missing_fields(invoices: list) -> list:
    """Flag invoices with empty required fields."""
    flags = []
    for inv in invoices:
        inv_id = inv.get("invoice_number", "UNKNOWN")
        missing = [f for f in REQUIRED_FIELDS if not inv.get(f)]
        if missing:
            flags.append({
                "invoice":  inv_id,
                "type":     "MISSING_FIELDS",
                "severity": "MEDIUM",
                "detail":   f"Missing fields: {', '.join(missing)}",
                "file":     inv.get("source_file", ""),
            })
    return flags


def check_amount_mismatch(invoices: list, pos: list) -> list:
    """Flag invoices where total exceeds PO approved amount by > 5%."""
    po_map = {p["po_id"]: p["approved_amount"] for p in pos}
    flags  = []
    for inv in invoices:
        po_ref  = inv.get("po_reference", "")
        total   = float(inv.get("total_amount") or 0)
        inv_id  = inv.get("invoice_number", "UNKNOWN")

        if po_ref and po_ref in po_map:
            approved = po_map[po_ref]
            if total > approved * (1 + OVERCHARGE_THRESHOLD):
                excess_pct = ((total - approved) / approved) * 100
                flags.append({
                    "invoice":  inv_id,
                    "type":     "AMOUNT_MISMATCH",
                    "severity": "HIGH",
                    "detail":   (
                        f"Invoice total ₹{total:,.2f} exceeds PO {po_ref} "
                        f"approved amount ₹{approved:,.2f} by {excess_pct:.1f}%."
                    ),
                    "file":     inv.get("source_file", ""),
                })
        elif po_ref and po_ref not in po_map:
            flags.append({
                "invoice":  inv_id,
                "type":     "INVALID_PO_REFERENCE",
                "severity": "MEDIUM",
                "detail":   f"PO reference {po_ref!r} not found in PO master.",
                "file":     inv.get("source_file", ""),
            })
    return flags


def check_gstin_format(invoices: list) -> list:
    """Basic GSTIN format validation (15 characters)."""
    flags = []
    for inv in invoices:
        gstin  = inv.get("vendor_gstin", "")
        inv_id = inv.get("invoice_number", "UNKNOWN")
        if gstin and len(str(gstin)) != 15:
            flags.append({
                "invoice":  inv_id,
                "type":     "INVALID_GSTIN",
                "severity": "LOW",
                "detail":   f"GSTIN {gstin!r} is not 15 characters.",
                "file":     inv.get("source_file", ""),
            })
    return flags


def run_all_rule_checks(invoices: list, pos: list) -> list:
    """Run all rule-based checks and return combined flags."""
    flags = []
    flags.extend(check_duplicates(invoices))
    flags.extend(check_missing_fields(invoices))
    flags.extend(check_amount_mismatch(invoices, pos))
    flags.extend(check_gstin_format(invoices))
    return flags


# ── LLM-powered audit summary ────────────────────────────────────────────────

AUDIT_PROMPT = PromptTemplate.from_template("""
You are a senior AP (Accounts Payable) audit manager reviewing a batch of invoices.
The automated system has flagged the following anomalies:

{flags_json}

For each anomaly provide:
1. Severity assessment (HIGH / MEDIUM / LOW)
2. Recommended action (HOLD / INVESTIGATE / APPROVE WITH NOTE / REJECT)
3. One-line business justification

Then provide a brief executive summary (3–4 sentences) of the overall batch health.

Format your response clearly with sections per anomaly, then the summary.
""")


def ai_audit_summary(flags: list) -> str:
    """Use Claude to generate a structured audit summary for detected anomalies."""
    if not flags:
        return (
            "✅ All invoices passed automated validation. "
            "No anomalies detected in this batch."
        )
    llm    = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)
    chain  = AUDIT_PROMPT | llm
    result = chain.invoke({"flags_json": json.dumps(flags, indent=2)})
    return result.content


# ── Runner ───────────────────────────────────────────────────────────────────

def run_detection(
    extracted_path: str = "data/extracted_invoices.json",
    pos_path:       str = "data/pos_meta.json",
    output_path:    str = "data/anomalies.json",
):
    if not os.path.exists(extracted_path):
        print(f"❌ {extracted_path} not found. Run batch_extract.py first.")
        return

    invoices = json.load(open(extracted_path))
    pos      = json.load(open(pos_path)) if os.path.exists(pos_path) else []

    print(f"Running anomaly checks on {len(invoices)} invoices...\n")
    flags = run_all_rule_checks(invoices, pos)

    # Print summary
    by_type = {}
    for f in flags:
        by_type.setdefault(f["type"], []).append(f)

    if flags:
        print(f"⚠️  Found {len(flags)} anomalies:\n")
        for flag_type, items in by_type.items():
            print(f"  [{flag_type}] × {len(items)}")
            for item in items:
                print(f"    → {item['invoice']}: {item['detail']}")
        print()
    else:
        print("✅ No anomalies detected.\n")

    # AI summary (traced automatically by LangSmith)
    print("Generating AI audit summary...")
    summary = ai_audit_summary(flags)
    print("\n── AI Audit Summary ──────────────────────────────────")
    print(summary)

    # Save
    output = {"flags": flags, "ai_summary": summary, "total_invoices": len(invoices)}
    json.dump(output, open(output_path, "w"), indent=2)
    print(f"\n✅ Anomaly report saved → {output_path}")
    print("Next step: streamlit run app.py")


if __name__ == "__main__":
    run_detection()
