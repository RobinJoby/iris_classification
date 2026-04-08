# model.py

import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")