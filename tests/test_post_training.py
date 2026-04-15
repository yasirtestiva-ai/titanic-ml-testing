import pandas as pd
import numpy as np
import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


#  preprocessing 
def preprocess():
    df = pd.read_csv("data/Titanic.csv")

    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    return df


# ── 1. Regression Test ───────────────────────────────────────
def test_regression():
    old_preds = pd.read_csv("data/test_predictions.csv")["Predicted"].values

    df = preprocess()
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    new_preds = model.predict(X_test)

    match_rate = np.mean(old_preds == new_preds)
    assert match_rate >= 0.90


# ── 2. Drift Detection ───────────────────────────────────────
def test_drift_detection():
    df = preprocess()
    X = df.drop(columns=["Survived"])

    reference = X.iloc[:600]
    current = X.iloc[600:].copy()

    current["Age"] += 15

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result = report.as_dict()["metrics"][0]["result"]

    assert result["share_of_drifted_columns"] < 0.50


# ── 3. Fairness Test ─────────────────────────────────────────
def test_fairness():
    df = pd.read_csv("data/test_predictions.csv")

    y_true = df["Actual"]
    y_pred = df["Predicted"]

    female = df["Sex"] == 0
    male = df["Sex"] == 1

    f1_female = f1_score(y_true[female], y_pred[female])
    f1_male = f1_score(y_true[male], y_pred[male])

    assert abs(f1_female - f1_male) < 0.15


# ── 4. Adversarial Test ──────────────────────────────────────
def test_adversarial():
    model = joblib.load("model/model.pkl")
    df = preprocess()

    sample = df[df["Pclass"] == 1].drop(columns=["Survived"]).head(20)

    base = model.predict(sample).mean()

    sample["Pclass"] = 3
    attacked = model.predict(sample).mean()

    assert attacked < base