import os
import fitz  # PyMuPDF
from utils.cleaning import basic_clean, tag_sections
from utils.io import write_jsonl

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"

os.makedirs(CLEAN_DIR, exist_ok=True)

records = []
for fn in os.listdir(RAW_DIR):
    if not fn.lower().endswith(".pdf"):
        continue
    doc_id = fn
    text_parts = []
    with fitz.open(os.path.join(RAW_DIR, fn)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    text = basic_clean("\n".join(text_parts))
    secs = tag_sections(text)
    with open(os.path.join(CLEAN_DIR, f"{fn}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    for s in secs:
        s["doc_id"] = doc_id
        records.append(s)

write_jsonl(os.path.join(CLEAN_DIR, "sections.jsonl"), records)
print(f"Saved {len(records)} sections → data/clean/sections.jsonl")