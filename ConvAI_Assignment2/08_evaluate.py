import os
import time
import importlib.util
from utils.io import read_jsonl
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.guardrails import validate_query

# Load test set (at least 10)
TEST = read_jsonl("data/qa/test.jsonl")[:10]

# Dynamically import 05_rag_generate.py (numeric filename not importable as a normal module)
rag_path = os.path.join(os.path.dirname(__file__), "05_rag_generate.py")
spec = importlib.util.spec_from_file_location("rag_mod", rag_path)
rag_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_mod)
rag_answer = rag_mod.answer

# Load FT model
ft_tok = AutoTokenizer.from_pretrained("models/ft_gen")
ft_mdl = AutoModelForCausalLM.from_pretrained("models/ft_gen")

def ft_generate(q):
    # ensure pad token
    if ft_tok.pad_token is None:
        ft_tok.pad_token = ft_tok.eos_token
    ft_mdl.config.pad_token_id = ft_tok.pad_token_id

    prompt = (
        "Instruction: Answer the financial question concisely.\n"
        "If unknown, say: Data not in scope.\n"
        f"Question: {q}\n"
        "Answer:"
    )
    enc = ft_tok(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True)
    out = ft_mdl.generate(
        **enc,
        max_new_tokens=64,
        do_sample=False,
        no_repeat_ngram_size=6,
        repetition_penalty=1.15,
        eos_token_id=ft_tok.eos_token_id,
        pad_token_id=ft_tok.pad_token_id,
    )
    raw = ft_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    import re
    ans = re.sub(r"(?is)question:.*?answer:\s*", "", raw).strip()
    ans = re.sub(r"\s+", " ", ans).strip()
    ans = ans.split("\n")[0]
    if not ans:
        ans = "Data not in scope"
    return {"answer": ans, "confidence": 0.75, "method": "FT"}


rows = [["Question","Method","Answer","Confidence","Time (s)","Correct (Y/N)"]]
for ex in TEST:
    q, gold = ex["question"], ex["answer"]
    # RAG
    t0 = time.perf_counter(); r = rag_answer(q); rt = time.perf_counter() - t0
    # FT
    t1 = time.perf_counter(); f = ft_generate(q); ft = time.perf_counter() - t1
    rows.append([q, "RAG", r["answer"], r.get("confidence", 0.0), round(rt, 2), int(r["answer"].lower() in gold.lower())])
    rows.append([q, "Fine‑Tune", f["answer"], f.get("confidence", 0.0), round(ft, 2), int(f["answer"].lower() in gold.lower())])

# --- replace the whole "write table" block in 08_evaluate.py with this ---
import csv, os

os.makedirs("reports/results", exist_ok=True)
with open("reports/results/compare.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Question","Method","Answer","Confidence","Time (s)","Correct (Y/N)"])
    for row in rows[1:]:  # rows[0] already has a header in our code; or just write from your list directly
        # Ensure no commas break columns (optional; csv.writer handles quoting anyway):
        safe = [str(x).replace("\n"," ").strip() for x in row]
        w.writerow(safe)
print("Saved → reports/results/compare.csv")
