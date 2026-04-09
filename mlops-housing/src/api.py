import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import uvicorn
import os
import logging
from datetime import datetime
import sqlite3
from fastapi import File, UploadFile
from sklearn.linear_model import LinearRegression


# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = joblib.load(model_path)

# Define input format
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Init FastAPI app
app = FastAPI(title="California Housing Predictor")

# Logging config
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Prometheus metric
prediction_counter = Counter("total_predictions", "Total prediction requests served")

# Ensure SQLite logs table exists
def init_db():
    conn = sqlite3.connect("logs/predictions.db")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS logs (
        timestamp TEXT,
        MedInc REAL, HouseAge REAL, AveRooms REAL, AveBedrms REAL,
        Population REAL, AveOccup REAL, Latitude REAL, Longitude REAL,
        prediction REAL
    )""")
    conn.commit()
    conn.close()

init_db()

@app.post("/predict")
def predict_price(input: HousingInput):
    df = pd.DataFrame([input.dict()])
    prediction = model.predict(df)[0]

    # Increment Prometheus counter
    prediction_counter.inc()

    # Log to file
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input": input.dict(),
        "prediction": round(prediction, 3)
    }
    logging.info(log_data)

    # Log to SQLite
    conn = sqlite3.connect("logs/predictions.db")
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (log_data["timestamp"], *df.iloc[0].tolist(), log_data["prediction"]))
    conn.commit()
    conn.close()

    return {"predicted_price": log_data["prediction"]}

@app.post("/retrain")
def retrain_model(file: UploadFile = File(...)):
    try:
        # Step 1: Load the new data from uploaded file
        new_data = pd.read_csv(file.file)

        # Step 2: Validate columns
        expected_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                         'Population', 'AveOccup', 'Latitude', 'Longitude', 'target']
        if not all(col in new_data.columns for col in expected_cols):
            return {"error": f"Invalid columns in uploaded file. Expected columns: {expected_cols}"}

        # Step 3: Load original training data
        original_data = pd.read_csv("data/housing.csv")  # Ensure this file exists in your repo

        # Step 4: Combine old + new data
        combined_data = pd.concat([original_data, new_data], ignore_index=True)

        X = combined_data.drop("target", axis=1)
        y = combined_data["target"]

        # Step 5: Retrain model
        new_model = LinearRegression()
        new_model.fit(X, y)

        # Step 6: Save updated model
        joblib.dump(new_model, model_path)

        # Step 7: Log retraining
        with open("logs/retraining.log", "a") as logf:
            logf.write(f"{datetime.now().isoformat()} - Retrained model on {len(combined_data)} rows "
                       f"(new rows: {len(new_data)}).\n")

        return {"message": "Model retrained and saved successfully."}

    except Exception as e:
        return {"error": str(e)}



# ✅ Prometheus metrics endpoint
@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ✅ Custom SQLite-based metrics endpoint
@app.get("/custom-metrics")
def custom_metrics():
    try:
        conn = sqlite3.connect("logs/predictions.db")
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
        if cursor.fetchone() is None:
            return {"error": "No logs found yet. Hit /predict at least once."}

        cursor.execute("SELECT COUNT(*) FROM logs")
        total_requests = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(prediction) FROM logs")
        avg_pred = cursor.fetchone()[0]

        conn.close()

        return {
            "total_requests": total_requests,
            "average_prediction": round(avg_pred, 3) if avg_pred else None
        }

    except Exception as e:
        return {"error": str(e)}

# Run app
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
