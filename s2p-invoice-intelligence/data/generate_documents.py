# data/generate_documents.py
# DAY 1 — Run this first to generate all synthetic S2P documents
# Usage: python data/generate_documents.py

from faker import Faker
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import random
import os
import json

fake = Faker("en_IN")  # Indian locale for realistic GST invoices
random.seed(42)        # reproducible output

OUTPUT_INVOICES = "data/invoices"
OUTPUT_POS      = "data/pos"
os.makedirs(OUTPUT_INVOICES, exist_ok=True)
os.makedirs(OUTPUT_POS,      exist_ok=True)

# ── Shared vendor pool ──────────────────────────────────────────────────────
def make_vendor():
    state_code = str(random.randint(10, 36))
    pan        = fake.bothify("?????#####?").upper()
    gstin      = f"{state_code}{pan}#Z#".replace("#", str(random.randint(1, 9)))
    return {
        "name":    fake.company(),
        "gstin":   gstin,
        "address": fake.address().replace("\n", ", "),
        "email":   fake.company_email(),
        "bank":    fake.bban(),
    }

VENDORS = [make_vendor() for _ in range(8)]

# ── PO generator ────────────────────────────────────────────────────────────
def generate_po(po_id: int, vendor: dict, amount: float) -> dict:
    path = f"{OUTPUT_POS}/PO-{po_id:04d}.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4

    # Header
    c.setFillColorRGB(0.12, 0.29, 0.49)
    c.rect(0, H - 80, W, 80, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, H - 45, "PURCHASE ORDER")
    c.setFont("Helvetica", 10)
    c.drawRightString(W - 50, H - 40, f"PO-{po_id:04d}")

    # Body
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, H - 110, "Vendor Details")
    c.setFont("Helvetica", 10)
    c.drawString(50, H - 128, f"Name    : {vendor['name']}")
    c.drawString(50, H - 144, f"GSTIN   : {vendor['gstin']}")
    c.drawString(50, H - 160, f"Address : {vendor['address'][:70]}")
    c.drawString(50, H - 176, f"Email   : {vendor['email']}")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, H - 210, "Order Details")
    c.setFont("Helvetica", 10)
    c.drawString(50, H - 228, f"PO Date          : {fake.date_this_year()}")
    terms = random.choice([30, 45, 60])
    c.drawString(50, H - 244, f"Payment Terms    : Net {terms} days")
    c.drawString(50, H - 260, f"Delivery By      : {fake.future_date()}")
    c.drawString(50, H - 276, f"Approved Amount  : INR {amount:,.2f}")
    c.drawString(50, H - 292, f"Approved By      : {fake.name()}")

    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, H - 330, "Description of Goods / Services")
    c.line(50, H - 335, W - 50, H - 335)
    c.setFont("Helvetica", 10)
    item = fake.bs().title()
    qty  = random.randint(1, 20)
    rate = round(amount / qty, 2)
    c.drawString(50,      H - 352, item[:50])
    c.drawString(380,     H - 352, f"Qty: {qty}")
    c.drawString(450,     H - 352, f"Rate: {rate:,.2f}")
    c.drawString(520,     H - 352, f"INR {amount:,.2f}")

    c.save()
    return {
        "po_id":            f"PO-{po_id:04d}",
        "vendor_name":      vendor["name"],
        "vendor_gstin":     vendor["gstin"],
        "approved_amount":  amount,
        "payment_terms":    terms,
        "file":             path,
    }

# ── Invoice generator ────────────────────────────────────────────────────────
def generate_invoice(inv_id: int, po_id: int, vendor: dict,
                     amount: float, anomaly: str = None) -> dict:
    path = f"{OUTPUT_INVOICES}/INV-{inv_id:04d}.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4

    # Header
    c.setFillColorRGB(0.08, 0.47, 0.34)
    c.rect(0, H - 80, W, 80, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, H - 45, "TAX INVOICE")
    c.setFont("Helvetica", 10)
    c.drawRightString(W - 50, H - 35, f"INV-{inv_id:04d}")
    c.drawRightString(W - 50, H - 52, f"Date: {fake.date_this_year()}")

    # Vendor block
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, H - 105, "FROM (Vendor)")
    c.setFont("Helvetica", 9)
    c.drawString(50, H - 120, vendor["name"])
    c.drawString(50, H - 133, f"GSTIN: {vendor['gstin']}")
    c.drawString(50, H - 146, vendor["address"][:60])
    c.drawString(50, H - 159, f"Bank A/C: {vendor['bank']}")

    # PO reference
    c.setFont("Helvetica-Bold", 10)
    c.drawString(320, H - 105, "PO Reference")
    c.setFont("Helvetica", 9)
    # anomaly: wrong PO reference
    po_ref = f"PO-{po_id:04d}" if anomaly != "wrong_po" else "PO-9999"
    c.drawString(320, H - 120, po_ref)
    c.drawString(320, H - 133, f"Payment Terms: Net {random.choice([30,45,60])}")

    # Line items table
    c.setFillColorRGB(0.08, 0.47, 0.34)
    c.rect(40, H - 230, W - 80, 22, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9)
    for x, label in [(50,"Description"),(330,"Qty"),(390,"Rate"),(470,"Amount")]:
        c.drawString(x, H - 222, label)

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 9)
    item  = fake.bs().title()
    qty   = random.randint(1, 20)
    rate  = round(amount / qty, 2)
    c.drawString(50,  H - 248, item[:45])
    c.drawString(330, H - 248, str(qty))
    c.drawString(390, H - 248, f"{rate:,.2f}")
    c.drawString(470, H - 248, f"{amount:,.2f}")

    # Totals
    cgst  = round(amount * 0.09, 2)
    sgst  = round(amount * 0.09, 2)
    total = amount + cgst + sgst

    c.line(40, H - 268, W - 40, H - 268)
    c.setFont("Helvetica", 9)
    for y_off, label, value in [
        (285, "Subtotal :", amount),
        (300, "CGST 9%  :", cgst),
        (315, "SGST 9%  :", sgst),
    ]:
        c.drawString(390, H - y_off, label)
        c.drawRightString(W - 50, H - y_off, f"{value:,.2f}")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(390, H - 335, "Total (INR):")
    c.drawRightString(W - 50, H - 335, f"{total:,.2f}")

    # Anomaly watermark (hidden metadata — for detection testing)
    if anomaly:
        c.setFont("Helvetica", 6)
        c.setFillColorRGB(0.95, 0.95, 0.95)
        c.drawString(50, 30, f"[TEST ANOMALY: {anomaly}]")

    c.save()
    return {
        "invoice_number": f"INV-{inv_id:04d}",
        "po_reference":   po_ref,
        "vendor_name":    vendor["name"],
        "vendor_gstin":   vendor["gstin"],
        "subtotal":       amount,
        "cgst":           cgst,
        "sgst":           sgst,
        "total_amount":   total,
        "anomaly":        anomaly,
        "file":           path,
    }

# ── Main generation ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Configure here ──────────────────────────────
    TOTAL_INVOICES   = 34    # change to however many you want
    NUM_OVERCHARGE   = 1     # invoices with amount > PO approved
    NUM_DUPLICATE    = 1     # invoices with duplicate invoice number
    NUM_WRONG_PO     = 1     # invoices with invalid PO reference
    # ────────────────────────────────────────────────

    NUM_ANOMALIES = NUM_OVERCHARGE + NUM_DUPLICATE + NUM_WRONG_PO
    NUM_NORMAL    = TOTAL_INVOICES - NUM_ANOMALIES

    print("Generating Purchase Orders...")
    pos = []
    for i in range(1, TOTAL_INVOICES + 1):
        vendor = VENDORS[i % len(VENDORS)]
        amount = round(random.uniform(5000, 100000), 2)
        pos.append(generate_po(i, vendor, amount))
    json.dump(pos, open("data/pos_meta.json", "w"), indent=2)
    print(f"  ✓ {len(pos)} POs saved to data/pos/")

    print("\nGenerating Invoices...")
    invoices = []

    # Normal invoices
    for i in range(1, NUM_NORMAL + 1):
        po     = pos[i - 1]
        vendor = next(v for v in VENDORS if v["name"] == po["vendor_name"])
        amount = round(po["approved_amount"] * random.uniform(0.85, 1.0), 2)
        invoices.append(generate_invoice(i, i, vendor, amount))

    # Anomaly: overcharge
    for j in range(NUM_OVERCHARGE):
        i      = NUM_NORMAL + j + 1
        po     = pos[i - 1]
        vendor = next(v for v in VENDORS if v["name"] == po["vendor_name"])
        amount = round(po["approved_amount"] * random.uniform(1.1, 1.3), 2)
        invoices.append(generate_invoice(i, i, vendor, amount, anomaly="overcharge"))

    # Anomaly: duplicate invoice number
    offset = NUM_NORMAL + NUM_OVERCHARGE
    for j in range(NUM_DUPLICATE):
        i      = offset + j + 1
        po     = pos[4]
        vendor = next(v for v in VENDORS if v["name"] == po["vendor_name"])
        dup    = generate_invoice(i, 5, vendor, pos[4]["approved_amount"] * 0.9)
        dup["invoice_number"] = "INV-0005"
        invoices.append(dup)

    # Anomaly: wrong PO reference
    offset = NUM_NORMAL + NUM_OVERCHARGE + NUM_DUPLICATE
    for j in range(NUM_WRONG_PO):
        i      = offset + j + 1
        po     = pos[i - 1]
        vendor = next(v for v in VENDORS if v["name"] == po["vendor_name"])
        amount = round(po["approved_amount"] * 0.9, 2)
        invoices.append(generate_invoice(i, i, vendor, amount, anomaly="wrong_po"))

    json.dump(invoices, open("data/invoices_meta.json", "w"), indent=2)
    print(f"  ✓ {TOTAL_INVOICES} invoices saved to data/invoices/")
    print("\n✅ All documents generated. Run extraction next!\n   python extraction/batch_extract.py")