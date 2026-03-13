import csv
import os
from collections import defaultdict

# ---------- Base directory ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERSON_DRUG_FILE = os.path.join(
    BASE_DIR, "data", "integrated", "person_drug.csv"
)

DRUG_SIDE_EFFECT_FILE = os.path.join(
    BASE_DIR, "data", "curated", "drug_side_effects_named.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "data", "final", "person_drug_side_effect.csv"
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ---------- Load drug -> side effects ----------
drug_to_side_effects = defaultdict(list)

with open(DRUG_SIDE_EFFECT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        drug = row["drug_name"].strip().lower()
        side_effect = row["side_effect"].strip()
        drug_to_side_effects[drug].append(side_effect)

print(f"[INFO] Loaded side effects for {len(drug_to_side_effects)} drugs")

# ---------- Stream person-drug file ----------
kept = 0
skipped = 0

with open(PERSON_DRUG_FILE, newline="", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)

    writer.writerow([
        "person_id",
        "drug_name",
        "side_effect",
        "genetic_score"
    ])

    for row in reader:
        person = row["Person"]
        drug = row["Drug"].strip().lower()
        score = row["Agg_Value"]

        if drug in drug_to_side_effects:
            for se in drug_to_side_effects[drug]:
                writer.writerow([person, drug, se, score])
                kept += 1
        else:
            skipped += 1

print("[DONE]")
print(f"Rows written : {kept}")
print(f"Skipped drugs: {skipped}")
print(f"Output saved to: {OUTPUT_FILE}")
