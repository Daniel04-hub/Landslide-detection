import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Create a comprehensive dataset with 4 risk classes
# Class 0: High Risk - Very dangerous conditions
# Class 1: Moderate Risk - Caution advised
# Class 2: Low Risk - Monitor conditions
# Class 3: Safe - No immediate danger

data = {
    'rainfall': [],
    'temperature': [],
    'soil_moisture': [],
    'slope': [],
    'humidity': [],
    'label': []
}

# High Risk samples (Class 0) - Extreme conditions
np.random.seed(42)
for _ in range(50):
    data['rainfall'].append(np.random.uniform(250, 500))  # Very heavy rainfall
    data['temperature'].append(np.random.uniform(10, 20))  # Cool temperatures
    data['soil_moisture'].append(np.random.uniform(80, 100))  # Very high moisture
    data['slope'].append(np.random.uniform(35, 60))  # Steep slopes
    data['humidity'].append(np.random.uniform(85, 100))  # High humidity
    data['label'].append(0)

# Moderate Risk samples (Class 1) - Caution conditions
for _ in range(50):
    data['rainfall'].append(np.random.uniform(150, 250))  # Moderate to heavy rainfall
    data['temperature'].append(np.random.uniform(20, 28))  # Moderate temperatures
    data['soil_moisture'].append(np.random.uniform(60, 80))  # High moisture
    data['slope'].append(np.random.uniform(25, 35))  # Moderate slopes
    data['humidity'].append(np.random.uniform(70, 85))  # Moderate to high humidity
    data['label'].append(1)

# Low Risk samples (Class 2) - Some risk but manageable
for _ in range(50):
    data['rainfall'].append(np.random.uniform(75, 150))  # Light to moderate rainfall
    data['temperature'].append(np.random.uniform(25, 32))  # Warmer temperatures
    data['soil_moisture'].append(np.random.uniform(40, 60))  # Moderate moisture
    data['slope'].append(np.random.uniform(15, 25))  # Gentle to moderate slopes
    data['humidity'].append(np.random.uniform(55, 70))  # Moderate humidity
    data['label'].append(2)

# Safe samples (Class 3) - Low risk conditions
for _ in range(50):
    data['rainfall'].append(np.random.uniform(0, 75))  # Little to no rainfall
    data['temperature'].append(np.random.uniform(28, 40))  # Warm to hot
    data['soil_moisture'].append(np.random.uniform(10, 40))  # Low moisture
    data['slope'].append(np.random.uniform(0, 15))  # Flat to gentle slopes
    data['humidity'].append(np.random.uniform(30, 55))  # Low to moderate humidity
    data['label'].append(3)

# Create DataFrame
df = pd.DataFrame(data)

# Save the new dataset
df.to_csv('land_slide.csv', index=False)
print("New dataset created with 4 risk classes!")
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['label'].value_counts().sort_index())

# Prepare features and labels
X = df[['rainfall', 'temperature', 'soil_moisture', 'slope', 'humidity']]
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['High Risk', 'Moderate Risk', 'Low Risk', 'Safe']))

# Test with the high-risk values from the user
test_high_risk = [[350, 15, 95, 45, 99]]
pred_high = model.predict(test_high_risk)[0]
prob_high = model.predict_proba(test_high_risk)[0]
print(f"\n{'='*60}")
print("Testing with HIGH RISK values: [350, 15, 95, 45, 99]")
print(f"Prediction: Class {pred_high}")
print("Probabilities:")
for i, prob in enumerate(prob_high):
    risk_names = ['High Risk', 'Moderate Risk', 'Low Risk', 'Safe']
    print(f"  {risk_names[i]}: {prob:.4f} ({prob*100:.2f}%)")
print(f"{'='*60}")

# Test with safe values
test_safe = [[20, 35, 15, 5, 40]]
pred_safe = model.predict(test_safe)[0]
prob_safe = model.predict_proba(test_safe)[0]
print(f"\nTesting with SAFE values: [20, 35, 15, 5, 40]")
print(f"Prediction: Class {pred_safe}")
print("Probabilities:")
for i, prob in enumerate(prob_safe):
    risk_names = ['High Risk', 'Moderate Risk', 'Low Risk', 'Safe']
    print(f"  {risk_names[i]}: {prob:.4f} ({prob*100:.2f}%)")
print(f"{'='*60}\n")

# Save the new model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ New model saved as 'model.pkl'")
print("✅ Dataset saved as 'land_slide.csv'")
print("\n🎉 Model retraining complete! The app should now work correctly.")
