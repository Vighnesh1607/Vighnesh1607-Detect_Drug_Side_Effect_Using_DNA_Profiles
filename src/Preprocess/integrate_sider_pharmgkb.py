import pandas as pd
import os

# Paths
SIDER_FILE = "data/curated/sider_clean.csv"
DRUG_MAP_FILE = "data/curated/drug_map_filtered.csv"
PHARMGKB_FILE = "data/curated/pharmgkb_with_drug_type.csv"

OUTPUT_DIR = "data/integrated"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gene_drug_side_effect.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load files
sider = pd.read_csv(SIDER_FILE)
drug_map = pd.read_csv(DRUG_MAP_FILE)
pharmgkb = pd.read_csv(PHARMGKB_FILE)

# Normalize text
drug_map["drug_name"] = drug_map["drug_name"].str.lower().str.strip()
pharmgkb["drug"] = pharmgkb["drug"].str.lower().str.strip()

# Step 1: SIDER + Drug Map
sider_drug = sider.merge(
    drug_map,
    on="drug_id",
    how="inner"
)

# Step 2: Join with PharmGKB
final_df = sider_drug.merge(
    pharmgkb,
    left_on="drug_name",
    right_on="drug",
    how="inner"
)

# Select required columns
final_df = final_df[[
    "gene",
    "drug_name",
    "drug_type",
    "side_effect"
]]

# Remove duplicates
final_df = final_df.drop_duplicates()

# Save final dataset
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ FINAL INTEGRATION COMPLETE")
print("📄 Total rows:", len(final_df))
print("📁 Saved at:", OUTPUT_FILE)
