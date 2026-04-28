# extraction/extractor.py
# DAY 2 — AI-powered multimodal invoice extraction using Claude Vision

import anthropic
import base64
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

EXTRACTION_PROMPT = """You are an expert AP (Accounts Payable) data extraction assistant.
Extract ALL fields from this invoice image and return ONLY a valid JSON object.
No explanation, no markdown, no backticks — raw JSON only.

Required JSON structure:
{
  "invoice_number": "",
  "po_reference": "",
  "invoice_date": "",
  "vendor_name": "",
  "vendor_gstin": "",
  "vendor_address": "",
  "line_items": [
    {
      "description": "",
      "quantity": 0,
      "unit_rate": 0.0,
      "amount": 0.0
    }
  ],
  "subtotal": 0.0,
  "cgst": 0.0,
  "sgst": 0.0,
  "igst": 0.0,
  "total_amount": 0.0,
  "payment_terms": "",
  "bank_account": "",
  "currency": "INR"
}

Rules:
- Use null for any field not found in the document
- All monetary values must be numbers (not strings)
- Extract EXACTLY what is written — do not infer or guess
"""


def pdf_to_base64_image(pdf_path: str) -> str:
    """Convert first page of a PDF to a base64-encoded PNG."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[0]
        # 150 DPI is enough for Claude to read text clearly
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.standard_b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"PDF conversion failed for {pdf_path}: {e}")


def extract_invoice_data(pdf_path: str) -> dict:
    """
    Extract structured data from an invoice PDF using Claude Vision.

    Args:
        pdf_path: Path to the invoice PDF file

    Returns:
        Dictionary with extracted invoice fields
    """
    img_b64 = pdf_to_base64_image(pdf_path)

    message = client.messages.create(
        model="claude-opus-4-5",   # Vision capability required
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT,
                    },
                ],
            }
        ],
    )

    raw_text = message.content[0].text.strip()

    # Strip accidental markdown fences
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    extracted = json.loads(raw_text)
    extracted["source_file"] = Path(pdf_path).name
    return extracted


if __name__ == "__main__":
    # Quick single-file test
    test_path = "data/invoices/INV-0001.pdf"
    if os.path.exists(test_path):
        print(f"Testing extraction on {test_path}...")
        result = extract_invoice_data(test_path)
        print(json.dumps(result, indent=2))
    else:
        print("Run data/generate_documents.py first.")
