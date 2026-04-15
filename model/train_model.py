import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('../data/data.csv')

# Rename columns
data = data.rename(columns={
    'moisture': 'soil_moisture',
    'temp': 'temperature',
    'pump': 'irrigation'
})

# Add humidity (fake for now)
data['humidity'] = np.random.randint(40, 80, size=len(data))

# Convert irrigation (1/0 → YES/NO)
def irrigation_logic(row):
    if row['soil_moisture'] > 600 and row['temperature'] > 30:
        return 'YES'
    elif row['soil_moisture'] > 500 and row['humidity'] < 50:
        return 'YES'
    else:
        return 'NO'

data['irrigation'] = data.apply(irrigation_logic, axis=1)

# Clean data
data = data.dropna()

# Features and label
X = data[['soil_moisture', 'temperature', 'humidity']]
y = data['irrigation']

# Convert YES/NO → 1/0
y = y.map({'YES': 1, 'NO': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Save model
with open('irrigation_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")