import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.retrieval import HybridRetriever
from utils.guardrails import validate_query, numeric_consistency

GEN_MODEL = "distilgpt2"
MAX_CTX = 800        # keep context short; tiny models copy if too long
TOP_K = 3            # fewer passages → better generation focus
ALPHA = 0.6

retr = HybridRetriever(); retr.load("indexes/faiss", "indexes/bm25")
tok = AutoTokenizer.from_pretrained(GEN_MODEL)
mdl = AutoModelForCausalLM.from_pretrained(GEN_MODEL)

# ensure pad token + no attention-mask warnings
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
mdl.config.pad_token_id = tok.pad_token_id

def build_prompt(q, passages):
    # Keep the labels, but instruct the model NOT to copy them
    ctx = "\n\n".join([f"[S{i+1}] {p}" for i, p in enumerate(passages)])
    return (
        "You are a precise finance assistant.\n"
        "Answer concisely using ONLY the facts in Sources. Do NOT copy Source labels.\n"
        "If the answer is not in Sources, say: Data not in scope.\n"
        "Format numbers as they appear (currency/symbols).\n\n"
        f"Sources:\n{ctx}\n\n"
        f"Question: {q}\n"
        "Answer:"
    )

def _fallback_extractive(q: str, passages: list[str]) -> str:
    """
    Simple extractive fallback: find the sentence in the retrieved context that
    best matches the query's main financial keyword(s) and includes a number.
    """
    text = " ".join(passages)
    # split to sentences
    sents = re.split(r'(?<=[.!?])\s+', text)
    # derive crude keywords from query
    key = q.lower()
    keywords = []
    if "revenue" in key: keywords += ["revenue", "total income", "turnover"]
    if "net profit" in key or "net income" in key: keywords += ["net profit", "profit for the year", "net income", "profit after tax", "pat"]
    if "operating cash" in key: keywords += ["net cash from operating activities", "operating cash"]
    if "equity" in key: keywords += ["equity", "shareholders' equity", "net worth"]
    if "debt" in key or "borrowings" in key: keywords += ["borrowings", "debt"]
    if not keywords:
        # fall back to any sentence with a number and a financial term
        keywords = ["revenue","income","profit","cash","equity","assets","liabilities","borrowings","debt"]

    NUM = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s?(?:%|crore|lakh|million|billion|₹|\$)?", re.I)
    scored = []
    for s in sents:
        low = s.lower()
        if any(k in low for k in keywords) and NUM.search(s):
            score = sum(k in low for k in keywords) + len(NUM.findall(s)) * 0.5
            scored.append((score, s))
    if not scored:
        return "Data not in scope"
    scored.sort(reverse=True)
    # return the best sentence trimmed
    return scored[0][1].strip()

def answer(q: str):
    ok, msg = validate_query(q)
    if not ok:
        return {"answer": "Blocked", "confidence": 0.0, "reason": msg, "method": "RAG"}

    hits = retr.search(q, top_k=TOP_K, alpha=ALPHA)
    passages = [h[1] for h in hits]
    prompt = build_prompt(q, passages)

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX, padding=True)
    out_ids = mdl.generate(
        **enc,
        max_new_tokens=96,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    raw = tok.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Guard: if it starts by echoing source labels or is empty, use extractive fallback
    if not raw or raw.startswith("[S") or raw.lower().startswith("sources") or raw.lower().startswith("question"):
        raw = _fallback_extractive(q, passages)

    # numeric-consistency confidence vs concatenated context
    ctx = "\n".join(passages)
    penalty = numeric_consistency(raw, ctx)
    retr_conf = sum([h[0] for h in hits]) / max(1, len(hits))
    conf = 0.5 * penalty + 0.5 * min(1.0, retr_conf)

    return {"answer": raw, "confidence": round(conf, 3), "sources": passages, "method": "RAG"}


if __name__ == "__main__":
    while True:
        try:
            q = input("Q> ")
            print(answer(q))
        except KeyboardInterrupt:
            break