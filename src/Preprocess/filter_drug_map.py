import pandas as pd
import os

# Paths
DRUG_MAP_FILE = "data/curated/drug_map_clean.csv"
PHARMGKB_FILE = "data/curated/pharmgkb_with_drug_type.csv"
OUTPUT_FILE = "data/curated/drug_map_filtered.csv"

os.makedirs("data/curated", exist_ok=True)

# Load data
drug_map = pd.read_csv(DRUG_MAP_FILE)
pharmgkb = pd.read_csv(PHARMGKB_FILE)

# Normalize text
drug_map["drug_name"] = drug_map["drug_name"].str.lower().str.strip()
pharmgkb["drug"] = pharmgkb["drug"].str.lower().str.strip()

# Valid drugs from PharmGKB
valid_drugs = set(pharmgkb["drug"].unique())

# Filter
filtered = drug_map[drug_map["drug_name"].isin(valid_drugs)]

# Remove duplicates
filtered = filtered.drop_duplicates(subset=["drug_id"])

# Save
filtered.to_csv(OUTPUT_FILE, index=False)

print("✅ drug_map filtered successfully")
print("Rows before:", len(drug_map))
print("Rows after :", len(filtered))
print("Saved to   :", OUTPUT_FILE)
