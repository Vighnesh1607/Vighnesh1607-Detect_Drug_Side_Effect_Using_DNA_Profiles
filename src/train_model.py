import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle

# Load processed data
df = pd.read_csv('data/final/processed_for_training.csv')

# Features and target
X = df[['drug_encoded', 'genetic_score']]
y = df['side_effect_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Unique classes: {len(y.unique())}")

# Train XGBoost model
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(y.unique()),
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
with open('src/trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to src/trained_model.pkl")