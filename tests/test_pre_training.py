import pandas as pd

df = pd.read_csv("data/Titanic.csv")


# Missing values check
def test_missing_values():
    important_columns = ["Age", "Fare", "Embarked", "Pclass", "Sex", "Survived"]

    for col in important_columns:
        missing_percent = df[col].isnull().sum() / len(df) * 100
        assert missing_percent < 30


# Schema: columns exist
def test_schema_columns_exist():
    expected_columns = ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]

    for col in expected_columns:
        assert col in df.columns


# Schema: data types
def test_schema_dtypes():
    assert df["Survived"].dtype in ["int64", "float64"]
    assert df["Age"].dtype in ["int64", "float64"]
    assert df["Fare"].dtype in ["int64", "float64"]
    assert df["Pclass"].dtype in ["int64", "float64"]


# Class imbalance
def test_class_imbalance():
    survival_counts = df["Survived"].value_counts(normalize=True)
    assert survival_counts.min() * 100 >= 30


# Data leakage
def test_no_data_leakage():
    suspicious_columns = ["rescue_time", "boat_number", "body_recovered", "death_confirmed"]

    for col in suspicious_columns:
        assert col not in df.columns


# Value checks
def test_age_range():
    assert df["Age"].dropna().between(0, 120).all()

def test_fare_range():
    assert (df["Fare"] >= 0).all()

def test_pclass_values():
    assert df["Pclass"].isin([1, 2, 3]).all()