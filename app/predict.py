import pickle
import numpy as np

# Load trained model
with open('../model/irrigation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prediction function
def predict_irrigation(soil_moisture, temperature, humidity):
    input_data = np.array([[soil_moisture, temperature, humidity]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "YES"
    else:
        return "NO"


# Test the function
if __name__ == "__main__":
    soil = float(input("Enter soil moisture: "))
    temp = float(input("Enter temperature: "))
    hum = float(input("Enter humidity: "))

    result = predict_irrigation(soil, temp, hum)
    print("Irrigation Needed:", result)