import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


def load_data(path="data/raw/housing.csv"):
    df = pd.read_csv(path)
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_and_log(model_name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = sqrt(mse)
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} -> RMSE: {rmse:.3f}, R²: {r2:.3f}")

        return model, rmse, r2


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("CaliforniaHousing")

    # Train both models and collect results
    model1, rmse1, r2_1 = train_and_log("LinearRegression", LinearRegression(), X_train, X_test, y_train, y_test)
    model2, rmse2, r2_2 = train_and_log("DecisionTree", DecisionTreeRegressor(max_depth=5), X_train, X_test, y_train, y_test)

    # Choose the better model (based on RMSE)
    if rmse1 < rmse2:
        best_model = model1
        print("✅ LinearRegression selected as best model")
    else:
        best_model = model2
        print("✅ DecisionTree selected as best model")

    # Save the best model to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.pkl")
    print("✅ Best model saved to models/model.pkl")
