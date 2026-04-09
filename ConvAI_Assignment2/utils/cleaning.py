import re
from typing import List

def basic_clean(text: str) -> str:
    # Remove page headers/footers and normalize whitespace
    text = re.sub(r"\n?\s*Page\s+\d+\s*\n", "\n", text, flags=re.I)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[\t\r\f]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

SECTION_PATTERNS = [
    ("income_statement", r"(?i)(consolidated )?statements? of (operations|income)"),
    ("balance_sheet", r"(?i)(consolidated )?balance sheets?"),
    ("cash_flow", r"(?i)statements? of cash flows?"),
    ("mdna", r"(?i)management'?s? discussion and analysis|MD&A"),
]

def tag_sections(text: str) -> List[dict]:
    lines = text.split("\n")
    tagged = []
    for i, line in enumerate(lines):
        for sec, pat in SECTION_PATTERNS:
            if re.search(pat, line):
                tagged.append({"section": sec, "line_no": i})
    chunks = []
    for j, meta in enumerate(tagged):
        start = meta["line_no"]
        end = tagged[j+1]["line_no"] if j+1 < len(tagged) else len(lines)
        chunks.append({
            "section": meta["section"],
            "text": "\n".join(lines[start:end]).strip(),
        })
    if not chunks:
        chunks = [{"section": "full_report", "text": text}]
    return chunks