import time

def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt

def simple_accuracy(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())