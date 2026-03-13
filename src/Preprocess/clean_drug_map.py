import pandas as pd
import os

# INPUT (your current file)
INPUT_FILE = "data/curated/drug_map.csv"

# OUTPUT (cleaned file for project)
OUTPUT_FILE = "data/curated/drug_map_clean.csv"

os.makedirs("data/curated", exist_ok=True)

# Read file
df = pd.read_csv(INPUT_FILE)

# Normalize drug_name column
df["drug_name"] = df["drug_name"].astype(str).str.strip().str.lower()

# Remove bad rows
df_clean = df[
    (df["drug_name"] != "error") &
    (df["drug_name"] != "not found") &
    (df["drug_name"] != "") &
    (~df["drug_name"].isna())
]

# Drop duplicates (very important)
df_clean = df_clean.drop_duplicates(subset=["drug_id"])

# Save cleaned file
df_clean.to_csv(OUTPUT_FILE, index=False)

print("✅ Drug map cleaned successfully")
print(f"📄 Rows before: {len(df)}")
print(f"📄 Rows after : {len(df_clean)}")
print(f"📁 Saved to  : {OUTPUT_FILE}")
