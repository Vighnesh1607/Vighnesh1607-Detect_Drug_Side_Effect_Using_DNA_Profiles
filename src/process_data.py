import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv('data/final/person_drug_side_effect.csv')

print("Data loaded, shape:", df.shape)

# Initialize encoders
drug_encoder = LabelEncoder()
side_effect_encoder = LabelEncoder()

# Encode categorical columns
df['drug_encoded'] = drug_encoder.fit_transform(df['drug_name'])
df['side_effect_encoded'] = side_effect_encoder.fit_transform(df['side_effect'])

print("Encoding done")

# Create new dataframe with encoded columns
processed_df = df[['drug_encoded', 'genetic_score', 'side_effect_encoded']]

# Save to new CSV
processed_df.to_csv('data/final/processed_for_training.csv', index=False)

print("Processed data saved to data/final/processed_for_training.csv")
print(f"Unique drugs: {len(drug_encoder.classes_)}")
print(f"Unique side effects: {len(side_effect_encoder.classes_)}")