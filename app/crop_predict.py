import pickle
import numpy as np

model = pickle.load(open("../model/crop_model.pkl", "rb"))

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)
    return prediction[0]