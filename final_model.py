import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load your dataset
df1 = pd.read_csv('land_slide.csv')

# Create a clean DataFrame using the correct column names
df = pd.DataFrame()

df['rainfall'] = df1['rainfall']
df['temperature'] = df1['temperature']
df['soil_moisture'] = df1['soil_moisture']
df['slope'] = df1['slope']
df['humidity'] = df1['humidity']
df['label'] = df1['label']

# Optional: Map label values (if your label column uses text)
# Example: if labels are "Yes"/"No", uncomment below
# df['label'] = df['label'].map({'No': 0, 'Yes': 1})

# Check for null values
print("Null values in dataset:\n", df.isnull().sum())

# Split dataset into features and target
X = df.drop("label", axis=1).values
y = df["label"].values

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Train RandomForestClassifier and tune estimators
error = []
for i in range(50, 100):  # reduced range for speed
    rfc = RandomForestClassifier(n_estimators=i, random_state=42)
    rfc.fit(X_train, y_train)
    pred_i = rfc.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Best model
rfc = RandomForestClassifier(n_estimators=75, random_state=42)
rfc.fit(X_train, y_train)

# Save model as pickle file
pickle.dump(rfc, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print("✅ Model training completed and saved successfully!")
