import csv
import os

PHARMGKB_FILE = "data/curated/pharmgkb_with_drug_type.csv"
SIDER_FILE = "data/curated/sider_clean.csv"
OUTPUT_FILE = "data/integrated/gene_drug_side_effect.csv"

os.makedirs("data/integrated", exist_ok=True)

# Load SIDER into memory (small enough)
drug_to_side_effects = {}

with open(SIDER_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        drug_id = row["drug"].lower()
        side_effect = row["side_effect"]

        drug_to_side_effects.setdefault(drug_id, set()).add(side_effect)

with open(PHARMGKB_FILE, "r", encoding="utf-8") as pharm_file, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as out_file:

    reader = csv.DictReader(pharm_file)
    writer = csv.writer(out_file)

    writer.writerow(["gene", "drug", "drug_type", "side_effect"])

    for row in reader:
        gene = row["gene"]
        drug = row["drug"]
        drug_type = row["drug_type"]

        # try to match drug name textually
        matched = False
        for sider_drug, side_effects in drug_to_side_effects.items():
            if drug.lower() in sider_drug:
                for se in side_effects:
                    writer.writerow([gene, drug, drug_type, se])
                matched = True

        # if no match found, skip safely
        if not matched:
            continue

print("Integration complete: gene–drug–side effect table created.")
