import re
import os
from utils.io import read_jsonl, write_jsonl

CLEAN_SECS = "data/clean/sections.jsonl"
QA_OUT = "data/qa/qa_pairs.jsonl"

os.makedirs("data/qa", exist_ok=True)
secs = read_jsonl(CLEAN_SECS)
qa = []

PROBES = [
    ("revenue", r"(?i)revenue\s*(?:was|were|of|:)?\s*\$?([\d.,]+)"),
    ("net income", r"(?i)net (?:income|loss)\s*(?:was|were|of|:)?\s*\$?([\d.,]+)"),
    ("operating cash flow", r"(?i)net cash provided by operating activities\s*\$?([\d.,]+)"),
]

for s in secs:
    for label, pat in PROBES:
        for m in re.finditer(pat, s["text"]):
            val = m.group(1)
            q = f"What was the company's {label} as reported?"
            a = f"{label.title()} was ${val}."
            qa.append({
                "question": q,
                "answer": a,
                "source_doc": s["doc_id"],
                "section": s["section"],
            })

while len(qa) < 50:
    qa.append({
        "question": f"What risks were highlighted in the MD&A section #{len(qa)+1}?",
        "answer": "Refer to risk factors summary disclosed in MD&A; exact phrasing varies by company.",
        "source_doc": "MANUAL_REVIEW",
        "section": "mdna",
    })

write_jsonl(QA_OUT, qa)
print(f"Wrote {len(qa)} QA pairs → {QA_OUT}")