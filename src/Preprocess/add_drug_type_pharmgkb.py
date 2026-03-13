import csv
import os

INPUT_FILE = "data/curated/pharmgkb_clean.csv"
OUTPUT_FILE = "data/curated/pharmgkb_with_drug_type.csv"

os.makedirs("data/curated", exist_ok=True)

GROUP_KEYWORDS = [
    "antibiotic",
    "antibiotics",
    "antifungal",
    "antifungals",
    "products",
    "agents",
    "preparations",
    "and"
]

def detect_drug_type(drug_name):
    name = drug_name.lower()
    for word in GROUP_KEYWORDS:
        if word in name:
            return "group"
    return "single"

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    writer.writerow(["gene", "drug", "drug_type", "evidence"])

    for row in reader:
        gene = row["gene"].strip()
        drug = row["drug"].strip()
        evidence = row["evidence"].strip()

        drug_type = detect_drug_type(drug)

        writer.writerow([gene, drug, drug_type, evidence])

print("Drug type added successfully.")
