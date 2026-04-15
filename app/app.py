from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model (correct path)
model = pickle.load(open("../model/irrigation_model.pkl", "rb"))

@app.route("/")
def home():
    return "AgriMind API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract inputs
        soil_moisture = data["soil_moisture"]
        temperature = data["temperature"]
        humidity = data["humidity"]

        # Convert to model input
        input_data = np.array([[soil_moisture, temperature, humidity]])

        # Prediction
        prediction = model.predict(input_data)[0]

        result = "YES" if prediction == 1 else "NO"

        return jsonify({"irrigation": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)