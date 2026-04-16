from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from crop_predict import predict_crop

app = Flask(__name__)
CORS(app)

# Load irrigation model
model = pickle.load(open("../model/irrigation_model.pkl", "rb"))

@app.route("/")
def home():
    return "AgriMind API is running 🚀"

# Irrigation prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        soil_moisture = data["soil_moisture"]
        temperature = data["temperature"]
        humidity = data["humidity"]

        input_data = np.array([[soil_moisture, temperature, humidity]])
        prediction = model.predict(input_data)[0]

        result = "YES" if prediction == 1 else "NO"

        return jsonify({"irrigation": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Crop prediction
@app.route('/predict_crop', methods=['POST'])
def crop_prediction():
    data = request.json

    result = predict_crop(
        data['N'],
        data['P'],
        data['K'],
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    )

    return jsonify({"recommended_crop": result})

# ✅ NEW: Combined prediction
@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.json

    # Crop prediction
    crop = predict_crop(
        data['N'],
        data['P'],
        data['K'],
        data['temperature'],
        data['humidity'],
        data['ph'],
        data['rainfall']
    )

    # Irrigation prediction
    irrigation_input = np.array([[
        data['soil_moisture'],
        data['temperature'],
        data['humidity']
    ]])

    irrigation_pred = model.predict(irrigation_input)[0]
    irrigation = "YES" if irrigation_pred == 1 else "NO"

    return jsonify({
        "recommended_crop": crop,
        "irrigation": irrigation
    })

# ALWAYS LAST
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)