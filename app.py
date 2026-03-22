from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Allow all origins (for frontend)
CORS(app)

PORT = int(os.getenv("PORT", 5001))

# Load model
model = joblib.load("student_model.pkl")


@app.route("/")
def home():
    return "ML Prediction API Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        hours = float(data["hours_studied"])
        attendance = float(data["attendance"])

        features = np.array([[hours, attendance]])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)