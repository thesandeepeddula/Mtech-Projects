import os
import sys
import importlib.util
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure project root is on sys.path so "utils.*" imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Dynamically import 05_rag_generate.py (filename starts with digits)
rag_path = os.path.join(ROOT, "05_rag_generate.py")
spec = importlib.util.spec_from_file_location("rag_mod", rag_path)
rag_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_mod)
rag_answer = rag_mod.answer


st.set_page_config(page_title="Financial QA: RAG vs Fine‑Tuning", layout="wide")
st.title("Financial QA: RAG vs Fine‑Tuning (Open‑Source)")
mode = st.radio("Mode", ["RAG", "Fine‑Tuned"], horizontal=True)
q = st.text_input("Ask a financial question about the uploaded reports:")


# Lazy‑load FT model
if "ft_tok" not in st.session_state:
    try:
        st.session_state.ft_tok = AutoTokenizer.from_pretrained("models/ft_gen")
        st.session_state.ft_mdl = AutoModelForCausalLM.from_pretrained("models/ft_gen")
    except Exception:
        st.session_state.ft_tok = None
        st.session_state.ft_mdl = None

if st.button("Answer") and q:
    if mode == "RAG":
        out = rag_answer(q)
        st.markdown(f"**Answer:** {out['answer']}")
        st.markdown(f"**Confidence:** {out.get('confidence',0.0):.2f}")
        with st.expander("Sources"):
            for i, s in enumerate(out.get('sources', [])[:3], 1):  # show top 3 for readability
                st.write(f"[Source {i}] {s[:700]}…")
        st.caption("Technique: Hybrid Search (BM25 + MiniLM)")
    else:
        if st.session_state.ft_tok is None:
            st.error("Fine-tuned model not found. Train it via 07_ft_train.py")
        else:
            tok = st.session_state.ft_tok
            mdl = st.session_state.ft_mdl

            # ensure pad token + consistent masks
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl.config.pad_token_id = tok.pad_token_id

            prompt = (
                "Instruction: Answer the financial question concisely.\n"
                "Do NOT repeat the question. If unknown, say: Data not in scope.\n"
                f"Question: {q}\n"
                "Answer:"
            )
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256, padding=True)
            out_ids = mdl.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,                   # deterministic (no loop drift)
                no_repeat_ngram_size=6,            # prevent phrase loops
                repetition_penalty=1.15,           # discourage repetition
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
            raw = tok.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

            # post-process: strip prompt-echoes + collapse whitespace
            import re
            ans = raw
            # remove any echoed Question/Answer headers
            ans = re.sub(r"(?is)question:.*?answer:\s*", "", ans).strip()
            ans = re.sub(r"\s+", " ", ans).strip()          # collapse weird newlines/spaces
            # hard stop after first sentence/line for crisp UI
            ans = ans.split("\n")[0].split("  ")[0].strip()

            if not ans:
                ans = "Data not in scope"

            st.markdown(f"**Answer:** {ans}")
            st.markdown("**Confidence:** 0.75 (heuristic)")
            st.caption("Technique: Adapter-based Fine-Tuning (LoRA)")


st.markdown("---")
st.subheader("How to run")
st.code("""\
python 01_preprocess.py
python 03_index.py
python 02_build_qa.py
python 06_ft_dataset.py
python 07_ft_train.py   # optional
python 08_evaluate.py
# then:
streamlit run app/streamlit_app.py
""", language="bash")