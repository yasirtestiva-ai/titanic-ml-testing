import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Load dataset
df = pd.read_csv("data/Titanic.csv")

# Drop unused columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical data
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

# Split features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

# Save predictions
X_test_copy = X_test.copy()
X_test_copy["Actual"] = y_test.values
X_test_copy["Predicted"] = y_pred

os.makedirs("data", exist_ok=True)
X_test_copy.to_csv("data/test_predictions.csv", index=False)

print("\nModel trained and results saved successfully")