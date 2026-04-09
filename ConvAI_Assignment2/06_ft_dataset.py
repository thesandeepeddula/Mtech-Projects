import random
import os
from utils.io import read_jsonl, write_jsonl

qa = read_jsonl("data/qa/qa_pairs.jsonl")
random.shuffle(qa)

n = len(qa)
train = qa[: int(0.8 * n)]
valid = qa[int(0.8 * n): int(0.9 * n)]
test  = qa[int(0.9 * n):]

os.makedirs("data/qa", exist_ok=True)
write_jsonl("data/qa/train.jsonl", train)
write_jsonl("data/qa/valid.jsonl", valid)
write_jsonl("data/qa/test.jsonl", test)
print({"train": len(train), "valid": len(valid), "test": len(test)})