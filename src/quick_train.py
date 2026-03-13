import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load a sample of processed data for quick training
df = pd.read_csv('data/final/processed_for_training.csv', nrows=10000)  # Sample 10k rows

X = df[['drug_encoded', 'genetic_score']]
y = df['side_effect_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use RandomForest for faster training
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save model
with open('src/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Quick model trained and saved.")