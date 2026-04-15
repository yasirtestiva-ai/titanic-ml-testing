import pandas as pd
import joblib
import pytest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data and model
df = pd.read_csv("data/test_predictions.csv")
model = joblib.load("model/model.pkl")

y_actual = df["Actual"]
y_pred = df["Predicted"]

#  preprocess Titanic data
def preprocess():
    raw = pd.read_csv("data/Titanic.csv")
    raw = raw.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    raw["Age"] = raw["Age"].fillna(raw["Age"].median())
    raw["Embarked"] = raw["Embarked"].fillna(raw["Embarked"].mode()[0])
    raw["Sex"] = LabelEncoder().fit_transform(raw["Sex"])
    raw["Embarked"] = LabelEncoder().fit_transform(raw["Embarked"])
    return raw.drop(columns=["Survived"]), raw["Survived"]


# 1. Accuracy test
def test_accuracy():
    assert accuracy_score(y_actual, y_pred) >= 0.75


# 2. F1 score test
def test_f1():
    assert f1_score(y_actual, y_pred) >= 0.70


# 3. Cross validation
def test_cross_validation():
    X, y = preprocess()
    model_cv = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model_cv, X, y, cv=5)
    assert scores.mean() >= 0.73


# 4. Confusion matrix check (FN rate)
def test_confusion_matrix():
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    fn_rate = fn / (fn + tp)
    assert fn_rate < 0.30


# 5. Error rate check
def test_error_rate():
    error_rate = (y_actual != y_pred).mean()
    assert error_rate < 0.30


# 6. Robustness tests
@pytest.mark.parametrize("input_data", [
    {"Pclass": 1, "Sex": 1, "Age": 999, "SibSp": 0, "Parch": 0, "Fare": 100, "Embarked": 1},
    {"Pclass": 3, "Sex": 0, "Age": 25, "SibSp": 0, "Parch": 0, "Fare": 0, "Embarked": 0}
])
def test_robustness(input_data):
    df_input = pd.DataFrame([input_data])
    pred = model.predict(df_input)
    assert pred[0] in [0, 1]