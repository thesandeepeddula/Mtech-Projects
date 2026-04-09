import re
from typing import Tuple

BLOCKLIST = ["hack", "password", "personal data", "ssn"]

def validate_query(q: str) -> Tuple[bool, str]:
    low = q.lower()
    for b in BLOCKLIST:
        if b in low:
            return False, f"Query blocked due to sensitive term: {b}"
    return True, "ok"

NUM_RE = re.compile(r"[-+]?(?:\d+[.,]?)+(?:%|\b)")

def numeric_consistency(answer: str, context: str) -> float:
    """Return a confidence penalty if numeric tokens in answer don't appear in context."""
    ans_nums = set(NUM_RE.findall(answer))
    ctx_nums = set(NUM_RE.findall(context))
    if not ans_nums:
        return 1.0
    overlap = ans_nums & ctx_nums
    return 1.0 if overlap == ans_nums else max(0.3, len(overlap) / max(1, len(ans_nums)))