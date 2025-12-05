# generate_model.py
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os

# ensure folder exists
os.makedirs("models", exist_ok=True)

# dummy dataset
X = np.array([
    [0.1, 0.2, 0.3],
    [0.7, 0.8, 0.9],
    [0.4, 0.5, 0.6]
])

y = np.array([0, 1, 1])

# train simple model
model = LogisticRegression()
model.fit(X, y)

# save it properly
joblib.dump(model, "models/model.joblib")

print("Model saved successfully!")
