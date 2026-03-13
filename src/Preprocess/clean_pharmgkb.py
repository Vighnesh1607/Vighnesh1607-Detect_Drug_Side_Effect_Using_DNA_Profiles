import csv
import os

# -------- Paths --------
INPUT_FILE = "data/pharmgkb/relationships.tsv"
OUTPUT_FILE = "data/curated/pharmgkb_clean.csv"

os.makedirs("data/curated", exist_ok=True)

# -------- Storage for unique gene-drug pairs --------
seen_pairs = set()

with open(INPUT_FILE, "r", encoding="utf-8") as tsv_file, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:

    reader = csv.DictReader(tsv_file, delimiter="\t")
    writer = csv.writer(csv_file)

    # write header
    writer.writerow(["gene", "drug", "evidence"])

    for row in reader:
        entity1_type = row["Entity1_type"].strip()
        entity2_type = row["Entity2_type"].strip()
        evidence = row["Evidence"].strip()

        # keep only Gene -> Chemical
        if entity1_type != "Gene" or entity2_type != "Chemical":
            continue

        # keep only strong evidence
        if not ("ClinicalAnnotation" in evidence or "VariantAnnotation" in evidence):
            continue

        gene = row["Entity1_name"].strip()
        drug = row["Entity2_name"].strip()

        key = (gene, drug)

        if key not in seen_pairs:
            seen_pairs.add(key)
            writer.writerow([gene, drug, evidence])

print(f"PharmGKB cleaning complete. Total unique gene-drug pairs: {len(seen_pairs)}")
