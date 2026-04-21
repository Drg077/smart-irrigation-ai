import pickle
import numpy as np

import os

model_path = os.path.join(os.path.dirname(__file__), "..", "model", "crop_model.pkl")
model = pickle.load(open(model_path, "rb"))

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    return prediction[0]