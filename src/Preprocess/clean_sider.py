import csv
import os

# -------- Paths --------
INPUT_FILE = "data/sider/sider.tsv"
OUTPUT_FILE = "data/curated/sider_clean.csv"

os.makedirs("data/curated", exist_ok=True)

# -------- Storage for unique pairs --------
seen_pairs = set()

with open(INPUT_FILE, "r", encoding="utf-8") as tsv_file, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:

    reader = csv.reader(tsv_file, delimiter="\t")
    writer = csv.writer(csv_file)

    # write header
    writer.writerow(["drug", "side_effect"])

    for row in reader:
        # safety check
        if len(row) < 6:
            continue

        drug_id = row[1].strip()          # PubChem / Drug identifier
        term_type = row[3].strip()        # LLT or PT
        side_effect = row[5].strip()

        # keep only PT terms
        if term_type != "PT":
            continue

        key = (drug_id, side_effect)

        if key not in seen_pairs:
            seen_pairs.add(key)
            writer.writerow([drug_id, side_effect])

print(f"SIDER cleaning complete. Total unique pairs: {len(seen_pairs)}")
