# extraction/batch_extract.py
# DAY 2 — Extract all invoices in batch and save to extracted_invoices.json
# Usage: python extraction/batch_extract.py

import json
import os
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from extraction.extractor import extract_invoice_data


def run_batch_extraction(invoice_dir: str = "data/invoices",
                         output_file: str = "data/extracted_invoices.json"):
    pdf_files = sorted(Path(invoice_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {invoice_dir}. Run generate_documents.py first.")
        return

    print(f"Found {len(pdf_files)} invoices to extract.\n")
    results = []
    failed  = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i:02d}/{len(pdf_files)}] Extracting {pdf_path.name}...", end=" ")
        try:
            data = extract_invoice_data(str(pdf_path))
            results.append(data)
            print(f"✓  Invoice: {data.get('invoice_number')} | "
                  f"Vendor: {data.get('vendor_name', 'N/A')[:30]} | "
                  f"Total: ₹{data.get('total_amount', 0):,.0f}")
        except Exception as e:
            print(f"✗  FAILED — {e}")
            failed.append({"file": pdf_path.name, "error": str(e)})

        # Small delay to respect API rate limits
        time.sleep(0.5)

    # Save results
    json.dump(results, open(output_file, "w"), indent=2)
    print(f"\n✅ Extracted {len(results)} invoices → {output_file}")

    if failed:
        fail_path = "data/extraction_failures.json"
        json.dump(failed, open(fail_path, "w"), indent=2)
        print(f"⚠️  {len(failed)} failures → {fail_path}")

    print("\nNext step: python rag/build_vectorstore.py")


if __name__ == "__main__":
    run_batch_extraction()
