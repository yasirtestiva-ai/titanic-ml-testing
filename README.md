```markdown
# Titanic ML Testing Project

A beginner-friendly Machine Learning Testing project built to practice and implement classical ML testing concepts — pre-training checks, model evaluation, and post-training testing — using the Titanic dataset.

---

## Project Goal

This is not a data science project. The goal is NOT to build the best model. The goal is to **test a machine learning model properly** — the way a real ML Testing Engineer would in a professional environment.

---

## Project Structure

```
titanic-ml-testing/
├── data/
│   ├── Titanic.csv              ← raw dataset
│   └── test_predictions.csv    ← saved predictions after training
├── model/
│   └── model.pkl               ← saved trained model
├── reports/
│   └── drift_report.html       ← evidently drift report
├── src/
│   └── train.py                ← trains and saves the model
├── tests/
│   ├── test_pre_training.py    ← data quality checks
│   ├── test_evaluation.py      ← model performance checks
│   └── test_post_training.py   ← post-deployment checks
└── requirements.txt
```

---

## Dataset

**Titanic Dataset** from Kaggle — 891 rows, binary classification (Survived: 0 or 1)

Key columns used:
- `Survived` — target variable (0 = died, 1 = survived)
- `Pclass` — ticket class (1st, 2nd, 3rd)
- `Sex` — male or female
- `Age` — passenger age (has missing values)
- `Fare` — ticket price
- `Embarked` — boarding port (has missing values)

---

## Model

**Logistic Regression** from scikit-learn. Simple, fast, interpretable. Trained with 80/20 train-test split. The model is kept simple on purpose — the focus is on testing, not model optimization.

---

## Testing Phases

### Phase 1 — Pre-Training Checks (`test_pre_training.py`)

Runs before training. Validates the dataset itself.

| Test | What it checks |
|---|---|
| `test_missing_values` | Critical columns have less than 30% missing values |
| `test_schema_columns_exist` | All expected columns are present in the file |
| `test_schema_dtypes` | Columns have correct data types (numeric where expected) |
| `test_class_imbalance` | Minority class is at least 30% of total data |
| `test_no_data_leakage` | No suspicious post-event columns exist |
| `test_age_range` | Age values are between 0 and 120 |
| `test_fare_range` | Fare values are 0 or above |
| `test_pclass_values` | Pclass contains only 1, 2, or 3 |

---

### Phase 2 — Model Evaluation (`test_evaluation.py`)

Runs after training. Validates model performance.

| Test | What it checks |
|---|---|
| `test_accuracy` | Model accuracy is at least 75% |
| `test_f1` | F1 score is at least 70% |
| `test_cross_validation` | Mean CV score across 5 folds is at least 73% |
| `test_confusion_matrix` | False negative rate is below 30% |
| `test_error_rate` | Overall error rate is below 30% |
| `test_robustness` | Model handles extreme/unusual inputs without crashing |

---

### Phase 3 — Post-Training Testing (`test_post_training.py`)

Simulates real-world post-deployment scenarios.

| Test | What it checks |
|---|---|
| `test_regression` | Retraining produces 90%+ consistent predictions |
| `test_drift_detection` | Less than 50% of features drift when new data shifts |
| `test_fairness` | F1 gap between male and female groups is acceptable |
| `test_adversarial` | Model reacts correctly when key features are manipulated |

> **Note on Fairness Test:** The fairness test detects a real bias — the model predicts female survival (F1: 0.88) significantly better than male survival (F1: 0.27). This reflects the historical "women and children first" pattern in the Titanic data. This is a genuine finding, not a code bug — and it demonstrates exactly what bias testing is designed to catch.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train.py
```

### 3. Run tests in order

```bash
# Pre-training checks
pytest tests/test_pre_training.py -v

# Model evaluation
pytest tests/test_evaluation.py -v

# Post-training testing
pytest tests/test_post_training.py -v
```

### 4. Run all tests at once
```bash
pytest tests/ -v
```

---

## Tools Used

| Tool | Purpose |
|---|---|
| pandas | Data loading, manipulation, checks |
| scikit-learn | Model training, metrics, cross-validation |
| joblib | Saving and loading the trained model |
| pytest | Running all tests as a proper test suite |
| evidently | Drift detection and reporting |
| matplotlib / seaborn | Visualizations (confusion matrix etc.) |

---

## Test Results Summary

| Phase | Tests | Status |
|---|---|---|
| Pre-Training | 8 tests | All Passed |
| Model Evaluation | 7 tests | All Passed |
| Post-Training | 4 tests | 3 Passed, 1 Failed (known bias) |

---

## Key Learnings

- Pre-training checks prevent training on bad data
- Evaluation metrics alone are not enough — cross-validation and error analysis matter
- Drift detection is critical for catching when real-world data shifts away from training data
- Bias testing can reveal real and explainable model behavior, not just bugs
- Regression testing ensures model updates don't silently break existing behavior

---

## Author

Built as part of an AI/ML Testing Engineer learning roadmap — Phase 1: Classical ML Model Testing.
```

Copy this into a `README.md` file in your project root. Clean, professional, and tells the full story of what you built and why. Ready for GitHub.
