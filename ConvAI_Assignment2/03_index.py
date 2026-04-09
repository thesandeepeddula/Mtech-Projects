from utils.io import read_jsonl, write_jsonl
from utils.chunking import make_chunk_records
from utils.retrieval import HybridRetriever

sections = read_jsonl("data/clean/sections.jsonl")
# explode into chunks at two sizes
records = []
for s in sections:
    records.extend(make_chunk_records(doc_id=s["doc_id"], section=s["section"], text=s["text"], sizes=(100, 400)))

write_jsonl("data/clean/chunks.jsonl", records)

retr = HybridRetriever()
retr.build(records, faiss_dir="indexes/faiss", bm25_dir="indexes/bm25")
print("Built hybrid indexes → indexes/faiss + indexes/bm25")