import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def load_and_save_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/housing.csv", index=False)
    print("✅ California housing data saved to data/raw/housing.csv")

if __name__ == "__main__":
    load_and_save_data()