
Here's a proper, well-formatted README:Now here's the actual markdown for your `README.md` file — copy this into your project:

```markdown
# titanic-ml-testing

![Python](https://img.shields.io/badge/Python-3.14-blue)
![pytest](https://img.shields.io/badge/tested%20with-pytest-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-purple)
![tests](https://img.shields.io/badge/tests-19-brightgreen)

Phase 1 of an AI/ML Testing Engineer roadmap — classical ML model testing using the Titanic dataset. Covers pre-training data checks, model evaluation, and post-training testing with pytest.

> This is not a data science project. The goal is not to build the best model — it's to **test a machine learning model properly**, the way a real ML Testing Engineer would.

---

## Project Structure

```
titanic-ml-testing/
├── data/
│   ├── Titanic.csv               # raw dataset (891 rows)
│   └── test_predictions.csv      # saved after training
├── model/
│   └── model.pkl                 # trained logistic regression
├── reports/
│   └── drift_report.html         # evidently drift output
├── src/
│   └── train.py                  # trains and saves model
├── tests/
│   ├── test_pre_training.py      # 8 data quality checks
│   ├── test_evaluation.py        # 7 model performance checks
│   └── test_post_training.py     # 4 post-deployment checks
└── requirements.txt
```

---

## Dataset

Titanic dataset from Kaggle — 891 rows, binary classification (Survived: 0 or 1).

Key columns: `Survived`, `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`

---

## How to Run

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train the model
python src/train.py

# 3. run each phase
pytest tests/test_pre_training.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_post_training.py -v

# or all at once
pytest tests/ -v
```

---

## Testing Phases

### Phase 1 — Pre-Training Checks
Validates the dataset before any training happens.

| Test | Checks |
|---|---|
| `test_missing_values` | critical columns < 30% null |
| `test_schema_columns_exist` | expected columns present |
| `test_schema_dtypes` | correct numeric types |
| `test_class_imbalance` | minority class ≥ 30% |
| `test_no_data_leakage` | no post-event columns |
| `test_age_range` | age between 0–120 |
| `test_fare_range` | fare ≥ 0 |
| `test_pclass_values` | only 1, 2, or 3 |

### Phase 2 — Model Evaluation
Validates model performance after training.

| Test | Checks |
|---|---|
| `test_accuracy` | accuracy ≥ 75% |
| `test_f1` | f1 score ≥ 70% |
| `test_cross_validation` | cv mean ≥ 73% (5-fold) |
| `test_confusion_matrix` | false negative rate < 30% |
| `test_error_rate` | overall error < 30% |
| `test_robustness` | handles extreme inputs without crashing |

### Phase 3 — Post-Training Testing
Simulates real-world post-deployment scenarios.

| Test | Checks |
|---|---|
| `test_regression` | retrain produces ≥ 90% same predictions |
| `test_drift_detection` | drifted columns < 50% |
| `test_fairness` | f1 gap between male and female groups |
| `test_adversarial` | pclass flip changes model prediction |

> **Note on Fairness:** `test_fairness` intentionally fails — F1 female (0.88) vs F1 male (0.27), gap of 0.61. This is a real bias finding correctly caught by the test. Historically explained by "women and children first" but flagged as expected model behavior.

---

## Test Results

| File | Result |
|---|---|
| `test_pre_training.py` | 8 / 8 passed |
| `test_evaluation.py` | 7 / 7 passed |
| `test_post_training.py` | 3 / 4 passed (known bias) |
| **Total** | **18 / 19 passed** |

---

## Tools Used

| Tool | Purpose |
|---|---|
| pandas | data loading and checks |
| scikit-learn | model training and metrics |
| pytest | test runner |
| evidently | drift detection |
| joblib | save and load model |

---

## Author

Built as Phase 1 of an AI/ML Testing Engineer learning roadmap.
```
