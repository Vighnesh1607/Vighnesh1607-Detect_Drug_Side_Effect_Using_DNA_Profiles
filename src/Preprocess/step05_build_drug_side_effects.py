import csv
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DRUG_MAP_FILE = os.path.join(BASE_DIR, "data", "curated", "drug_map_filtered.csv")
SIDER_FILE = os.path.join(BASE_DIR, "data", "curated", "sider_clean.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "curated", "drug_side_effects_named.csv")

# ---------- Load drug_id -> drug_name ----------
drug_id_to_name = {}

with open(DRUG_MAP_FILE, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        drug_id_to_name[row["drug_id"]] = row["drug_name"]

print(f"Loaded {len(drug_id_to_name)} drug IDs")

# ---------- Stream SIDER and write output ----------
with open(SIDER_FILE, newline='', encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)

    writer.writerow(["drug_name", "side_effect"])

    kept = 0
    skipped = 0

    for row in reader:
        drug_id = row["drug_id"]
        side_effect = row["side_effect"]

        if drug_id in drug_id_to_name:
            writer.writerow([drug_id_to_name[drug_id], side_effect])
            kept += 1
        else:
            skipped += 1

print(f"Done ✔  kept={kept}  skipped={skipped}")
print(f"Output saved to: {OUTPUT_FILE}")
