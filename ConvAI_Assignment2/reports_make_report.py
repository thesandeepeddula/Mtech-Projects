from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import csv, os

os.makedirs("reports/results", exist_ok=True)
path = "reports/results/summary.pdf"

c = canvas.Canvas(path, pagesize=A4)
W, H = A4
c.setFont("Helvetica-Bold", 16)
c.drawString(2*cm, H-2*cm, "RAG vs Fine‑Tuning – Comparison Summary")

c.setFont("Helvetica", 10)
y = H-3*cm
c.drawString(2*cm, y, "Table: Question | Method | Answer | Confidence | Time | Correct")
y -= 0.7*cm

with open("reports/results/compare.csv", encoding='utf-8') as f:
    rd = csv.reader(f)
    header = next(rd, None)
    for row in rd:
        line = " | ".join([row[0][:35] + ('…' if len(row[0])>35 else ''), row[1], row[2][:20], row[3], row[4], row[5]])
        c.drawString(2*cm, y, line)
        y -= 0.55*cm
        if y < 2*cm:
            c.showPage(); y = H-2*cm

c.save(); print(f"Saved → {path}")