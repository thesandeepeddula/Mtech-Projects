# test_retrain.py
import argparse
import csv
import random
import requests
from pathlib import Path

def make_fake_csv(path: Path, n_rows: int = 5):
    headers = [
        "MedInc","HouseAge","AveRooms","AveBedrms",
        "Population","AveOccup","Latitude","Longitude","target"
    ]
    rng = random.Random(42)
    rows = []
    for _ in range(n_rows):
        row = {
            "MedInc":      round(rng.uniform(0.5, 12.0), 4),
            "HouseAge":    round(rng.uniform(1, 52), 1),
            "AveRooms":    round(rng.uniform(2.0, 10.0), 3),
            "AveBedrms":   round(rng.uniform(0.5, 3.0), 3),
            "Population":  round(rng.uniform(100, 4000), 0),
            "AveOccup":    round(rng.uniform(1.0, 5.0), 3),
            # California-ish bounds (FastAPI validation-friendly if you add ranges later)
            "Latitude":    round(rng.uniform(32.5, 41.9), 4),
            "Longitude":   round(rng.uniform(-124.4, -114.1), 4),
            # target ~ MedHouseVal (in 100k’s, a rough synthetic mapping)
            "target":      round(rng.uniform(0.5, 5.5), 3)
        }
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Test /retrain endpoint with a generated CSV.")
    parser.add_argument("--url", default="http://localhost:8000/retrain", help="Retrain endpoint URL")
    parser.add_argument("--rows", type=int, default=5, help="Number of synthetic rows to generate")
    parser.add_argument("--outfile", default="new_data.csv", help="Where to write the CSV before upload")
    args = parser.parse_args()

    csv_path = Path(args.outfile).resolve()
    make_fake_csv(csv_path, n_rows=args.rows)
    print(f"✅ generated {csv_path} with {args.rows} rows")

    with csv_path.open("rb") as f:
        files = {"file": (csv_path.name, f, "text/csv")}
        try:
            resp = requests.post(args.url, files=files, timeout=30)
            print(f"🔁 POST {args.url} -> {resp.status_code}")
            try:
                print("↩️  response JSON:", resp.json())
            except Exception:
                print("↩️  response text:", resp.text)
        except requests.RequestException as e:
            print("❌ request error:", e)

if __name__ == "__main__":
    main()
