
# 🏠 California Housing Price Predictor - MLOps Pipeline

This project demonstrates a complete MLOps pipeline using the **California Housing Dataset**. The pipeline covers all stages from development to deployment, with versioning, experiment tracking, Dockerization, CI/CD, and monitoring.

---

## 📌 Project Team - Group 67

1. Sandeep Eddula - 2023AC05558
2. Omkar Manoj Kumar Dubal - 2023AB05156
3. Debanjali Dey - 2023AD05073
4. Akash Deep Bhattacharya - 2023AD05104



## 📌 Project Overview

- **Model Type**: Regression (Linear Regression & Decision Tree Regressor)
- **Dataset**: [California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- **Frameworks**: Scikit-learn, FastAPI, Docker, GitHub Actions, MLflow

---



## 📁 Project Structure

```
mlops-housing/
│
├── data/ # Dataset CSV
├── models/ # Trained models
├── logs/ # Logs & SQLite DB
├── src/
│ ├── train.py # Model training + MLflow logging
│ ├── evaluate.py # Model evaluation
│ ├── api.py # FastAPI app (prediction, metrics, retrain)
│ ├── test_retrain.py # Test script for retraining endpoint
│
├── prometheus.yml # Prometheus scrape config
├── Dockerfile # API Dockerfile
├── docker-compose.yml # Prometheus + Grafana setup
├── requirements.txt # Dependencies
├── .github/workflows/ci.yml # GitHub Actions pipeline
└── README.md # Project documentation
```

---

## ⚙️ Part 1: Data Versioning

- GitHub repo initialized and connected.
- Dataset stored in `data/housing.csv`
- DVC initialized for dataset tracking.

---

## 🧪 Part 2: Model Development

- Trained two models: `LinearRegression` and `DecisionTreeRegressor`
- Tracked experiments using MLflow:
  - Metrics: RMSE, R²
  - Parameters: max_depth, random_state
- Best model auto-logged and saved to `models/model.pkl`

---

## 🌐 Part 3: API Development & Dockerization

- FastAPI app with:
  - `/predict` → Predict house price
  - `/metrics` → Prometheus metrics
  - `/custom-metrics` → SQLite-based aggregated metrics
  - `/retrain` → Upload new CSV → retrain model

- Build & run:
```bash
docker build -t housing-predictor .
docker run -p 8000:8000 housing-predictor
  ```

---

## 🔁 Part 4: CI/CD with GitHub Actions

- On every push:
  - Lints Python code
  - Builds Docker image
  - Pushes to Docker Hub
- Secrets used:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`

---

## 📈 Part 5: Logging & Monitoring

- Logs all predictions to `logs/predictions.db`
- `/metrics` endpoint shows total predictions served
- SQLite used for persistent logging
- Optionally view `predictions.log`

---

## 🧪 Example Usage

### 🚀 POST Prediction
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d @sample.json
```

### 📊 GET Metrics
```bash
curl http://localhost:8000/metrics
```

---

🎯 Bonus Features
✅ Input validation via Pydantic

✅ Prometheus integration for monitoring

✅ Grafana dashboards

✅ Model retraining via /retrain endpoint

✅ Test script (test_retrain.py) auto-generates CSV and tests retraining

---

## 🎥 Demo Recording

> 📹 See `demo.mp4` for 5-min walkthrough


---

## 📅 Last Updated
**August 10, 2025**

---

## 👨‍💻 Author
[SandeepEddula](https://github.com/SandeepAmruta)
