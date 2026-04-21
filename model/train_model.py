import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
data = pd.read_csv('../data/data.csv')

# Rename columns
data = data.rename(columns={
    'moisture': 'soil_moisture',
    'temp': 'temperature'
})

# Add realistic humidity (if not present)
if 'humidity' not in data.columns:
    data['humidity'] = np.random.randint(30, 90, size=len(data))

# -------- SCALE SOIL MOISTURE TO SENSOR RANGE --------
# Normalize to 0–4095
data['soil_moisture'] = (data['soil_moisture'] / data['soil_moisture'].max()) * 4095

# Invert (sensor logic: 0=wet, 4095=dry)
data['soil_moisture'] = 4095 - data['soil_moisture']

# -------- SMART IRRIGATION LOGIC --------
def irrigation_logic(row):
    soil = row['soil_moisture']
    temp = row['temperature']
    hum  = row['humidity']

    # Very dry → always irrigate
    if soil > 3000:
        return 'YES'

    # Medium dry + high temperature
    elif soil > 2000 and temp > 30:
        return 'YES'

    # Medium dry + low humidity
    elif soil > 2000 and hum < 40:
        return 'YES'

    # Otherwise no irrigation
    else:
        return 'NO'

# Apply logic
data['irrigation'] = data.apply(irrigation_logic, axis=1)

# Clean data
data = data.dropna()

# Features and label
X = data[['soil_moisture', 'temperature', 'humidity']]
y = data['irrigation'].map({'YES': 1, 'NO': 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", model.score(X_test, y_test))
print(classification_report(y_test, y_pred))

# Save model
with open('../model/irrigation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained & saved successfully!")