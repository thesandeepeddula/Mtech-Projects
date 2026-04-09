from typing import List, Dict

def sliding_chunks(text: str, chunk_tokens: int = 400, stride_tokens: int = 60) -> List[str]:
    toks = text.split()
    out = []
    if chunk_tokens <= 0:
        return out
    step = max(1, chunk_tokens - stride_tokens)
    for i in range(0, len(toks), step):
        out.append(" ".join(toks[i:i+chunk_tokens]))
        if i + chunk_tokens >= len(toks):
            break
    return out

def make_chunk_records(doc_id: str, section: str, text: str, sizes=(100, 400)) -> List[Dict]:
    recs = []
    for sz in sizes:
        for ch in sliding_chunks(text, chunk_tokens=sz):
            recs.append({
                "doc_id": doc_id,
                "section": section,
                "chunk_size": sz,
                "text": ch,
            })
    return recs