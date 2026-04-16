
# 🚢 Titanic ML Testing Pipeline

A complete **Machine Learning Testing Pipeline** using the Titanic dataset.
This project focuses on **testing and validating ML systems** — not just training a model.

---

# 📌 Project Goal

The goal of this project is to ensure that a machine learning model is:

* ✅ Trained on clean and valid data
* ✅ Performing well (accuracy, F1 score, etc.)
* ✅ Stable over time
* ✅ Fair (not biased)
* ✅ Robust to unusual or manipulated inputs

👉 In short: **Build trust in ML systems before using them in real life**

---

# 🧠 Key Concepts Covered

* Data validation (before training)
* Model evaluation metrics
* Cross-validation
* Regression testing (ML consistency)
* Data drift detection
* Fairness testing
* Adversarial robustness

---

# 📁 Project Structure

```bash
titanic_ml_testing/
│
├── data/
│   ├── Titanic.csv
│   └── test_predictions.csv
│
├── model/
│   └── model.pkl
│
├── src/
│   └── train.py
│
├── tests/
│   ├── test_pretraining.py
│   ├── test_evaluation.py
│   └── test_post_training.py
│
└── README.md
```

---

# ⚙️ How It Works (Step-by-Step)

This project follows a **real-world ML pipeline**:

👉 **Step 1 → Check Data**
👉 **Step 2 → Train Model**
👉 **Step 3 → Evaluate Model**
👉 **Step 4 → Test in Real-World Conditions**

---

## 🔹 Step 1: Data Checking (Pre-Training)

📁 File: `test_pretraining.py`

Before training, we make sure the data is correct.

### What we check:

* Missing values are not too high
* Required columns exist
* Data types are correct
* No invalid values (e.g., Age > 120)
* No data leakage columns
* Data is not highly imbalanced

💡 **Why this matters:**
Bad data → bad model

---

## 🔹 Step 2: Model Training

📁 File: `src/train.py`

### What happens:

1. Load Titanic dataset
2. Remove unnecessary columns
3. Fill missing values
4. Convert text → numbers (encoding)
5. Split data:

   * Training set (80%)
   * Testing set (20%)
6. Train model (Logistic Regression)
7. Make predictions
8. Save:

   * Model → `model/model.pkl`
   * Predictions → `data/test_predictions.csv`

💡 **Why this matters:**
This is where the model learns patterns from data

---

## 🔹 Step 3: Model Evaluation

📁 File: `test_evaluation.py`

Now we check if the model is actually good.

### Tests included:

### ✅ Accuracy

* Percentage of correct predictions

### ✅ F1 Score

* Balance between precision and recall

### ✅ Cross Validation

* Model performance across multiple data splits

### ✅ Error Rate

* Percentage of wrong predictions

### ✅ Confusion Matrix (FN Check)

* Checks critical mistakes (e.g., predicting dead when survived)

### ✅ Robustness

* Model handles edge cases (like Age = 999)

💡 **Why this matters:**
Ensures model is reliable and not just lucky

---

## 🔹 Step 4: Real-World Testing (Post-Training)

📁 File: `test_post_training.py`

Now we simulate real-world problems.

---

### 🔁 Regression Test (Consistency)

* Compare old vs new predictions

👉 Ensures:

* Model behavior does not change unexpectedly

---

### 📉 Data Drift Detection

* Check if new data is different from old data

👉 Example:

* Age suddenly increases in new data

👉 If drift is high:

* Model may fail

---

### ⚖️ Fairness Test

* Compare model performance for:

  * Male vs Female

👉 Ensures:

* Model is not biased

---

### 🛡️ Adversarial Test

* Try to manipulate inputs

👉 Example:

* Change passenger class

👉 Ensures:

* Model reacts logically

---

# 🔥 Final Flow

```bash
Data → Validate → Train → Evaluate → Real-World Testing
```

---

# 🚀 How to Run

## 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib pytest evidently
```

---

## 2. Train the Model

```bash
python src/train.py
```

---

## 3. Run Tests

```bash
pytest
```

---

# 📊 Output

After running:

* Model performance printed in terminal
* Files generated:

  * `model/model.pkl`
  * `data/test_predictions.csv`

---

# 🧪 Testing Coverage Summary

| Stage         | What is Tested                         |
| ------------- | -------------------------------------- |
| Pre-training  | Data quality & validation              |
| Evaluation    | Model performance                      |
| Post-training | Stability, drift, fairness, robustness |

---

# 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Pytest
* Evidently

---

# 🔥 Why This Project Is Important

This project shows:

* ML Testing Engineering skills
* MLOps understanding
* Real-world ML system validation
* Responsible AI practices

👉 This is what companies expect beyond just training models

---

# 👨‍💻 Author

**Yasir Wali**
