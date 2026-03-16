import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")

X = data[["hours_studied", "attendance"]]
y = data["pass"]

model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("Model trained and saved!")