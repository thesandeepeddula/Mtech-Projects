from utils.retrieval import HybridRetriever

retr = HybridRetriever()
retr.load("indexes/faiss", "indexes/bm25")

while True:
    try:
        q = input("Query > ")
        if not q:
            continue
        hits = retr.search(q, top_k=5, alpha=0.6)
        for i, (sc, txt, meta) in enumerate(hits, 1):
            print(f"[{i}] score={sc:.3f} | {meta['doc_id']} | {meta['section']}\n{txt[:300]}...\n")
    except KeyboardInterrupt:
        break