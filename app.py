# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
iris = load_iris()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        prediction = model.predict(final_features)
        flower = iris.target_names[prediction][0]

        return render_template("index.html", prediction_text=f"Predicted Flower: {flower}")

    except:
        return render_template("index.html", prediction_text="Error in input!")

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))