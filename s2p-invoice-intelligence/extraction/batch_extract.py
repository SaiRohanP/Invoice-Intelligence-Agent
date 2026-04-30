# extraction/batch_extract.py
# DAY 2 — Extract all invoices in batch and save to extracted_invoices.json
# Usage: python extraction/batch_extract.py
#
# No API calls — uses local PyMuPDF text extraction, runs instantly.
# Resume support: already-extracted invoices are skipped on re-run.

import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from extraction.extractor import extract_invoice_data


def run_batch_extraction(
    invoice_dir: str = "data/invoices",
    output_file: str = "data/extracted_invoices.json",
):
    pdf_files = sorted(Path(invoice_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {invoice_dir}. Run generate_documents.py first.")
        return

    # ── Resume support ────────────────────────────────────────────────────────
    if Path(output_file).exists():
        existing     = json.load(open(output_file))
        already_done = {r["source_file"] for r in existing}
        if already_done:
            print(f"Resuming: {len(already_done)} invoice(s) already extracted, skipping.\n")
    else:
        existing     = []
        already_done = set()

    pending = [p for p in pdf_files if p.name not in already_done]
    total   = len(pdf_files)

    if not pending:
        print("All invoices already extracted. Nothing to do.")
        return

    print(f"Found {total} invoice(s) total — extracting {len(pending)} now.\n")

    done_count = total - len(pending)
    results    = list(existing)
    failed     = []

    for i, pdf_path in enumerate(pending, done_count + 1):
        print(f"[{i:02d}/{total}] Extracting {pdf_path.name}...", end=" ", flush=True)
        try:
            data = extract_invoice_data(str(pdf_path))
            results.append(data)
            print(
                f"OK  Invoice: {data.get('invoice_number')} | "
                f"Vendor: {(data.get('vendor_name') or 'N/A')[:30]} | "
                f"Total: {float(data.get('total_amount') or 0):,.0f}"
            )
        except Exception as e:
            print(f"FAILED — {e}")
            failed.append({"file": pdf_path.name, "error": str(e)})

        # Save after every file so a crash never loses work
        json.dump(results, open(output_file, "w"), indent=2)

    print(f"\nExtracted {len(results)} invoice(s) -> {output_file}")

    if failed:
        fail_path = "data/extraction_failures.json"
        json.dump(failed, open(fail_path, "w"), indent=2)
        print(f"  {len(failed)} failure(s) -> {fail_path}")

    print("\nNext step: python rag/build_vectorstore.py")


if __name__ == "__main__":
    run_batch_extraction()