# Titanic ML Testing Project

End-to-end Machine Learning testing project using the Titanic dataset.  
Focus is on **testing the ML pipeline**, not building a complex model.

---

## What I Built

- Preprocessed Titanic dataset
- Trained a Logistic Regression model
- Saved model and predictions
- Implemented **3 levels of ML testing**:
  - Data validation (before training)
  - Model evaluation (after training)
  - Post-training / production checks

---

## Project Structure

titanic-ml-testing/
│
├── data/
│   ├── Titanic.csv
│   └── test_predictions.csv
│
├── model/
│   └── model.pkl
│
├── reports/
│   └── drift_report.html
│
├── src/
│   └── train.py
│
├── tests/
│   ├── test_pre_training.py
│   ├── test_evaluation.py
│   └── test_post_training.py
│
└── requirements.txt

---

## Model

- Logistic Regression (scikit-learn)
- Train/Test Split: 80/20
- Simple model used intentionally (focus = testing)

---

## Testing Breakdown

### 1. Pre-Training (Data Validation)

- Missing values check  
- Column/schema validation  
- Data type validation  
- Class imbalance check  
- Data leakage check  
- Range checks (Age, Fare, Pclass)  

**Purpose:** Ensure clean and valid data before training  

---

### 2. Model Evaluation

- Accuracy ≥ 75%  
- F1 Score ≥ 70%  
- Cross-validation (5-fold)  
- Confusion matrix (low FN rate)  
- Error rate check  
- Robustness testing (edge inputs)  

**Purpose:** Ensure model performance is reliable  

---

### 3. Post-Training (Production Checks)

- Regression testing (prediction consistency)  
- Data drift detection (Evidently)  
- Fairness testing (male vs female)  
- Adversarial testing (feature manipulation)  

**Purpose:** Ensure model behaves correctly in real-world scenarios  

---

## Key Finding

Model performs better for females than males.  
This reflects the real Titanic pattern ("women and children first").  

This is a **detected bias**, not a bug.

---

## How to Run

Install dependencies:

pip install -r requirements.txt

Train model:

python src/train.py

Run all tests:

pytest tests/ -v

---

## Tools Used

- pandas  
- scikit-learn  
- pytest  
- joblib  
- evidently  

---

## What I Learned

- How to test ML systems end-to-end  
- Importance of data validation  
- Why accuracy alone is not enough  
- How to detect bias and drift  
- Writing automated ML tests using pytest  

---

## Author

Yasir Wali  
AI/ML Testing Project
