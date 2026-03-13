import csv
import os
from collections import defaultdict

# -------------------------------------------------
# Path handling (project-structured)
# -------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

GENE_MATRIX_FILE = os.path.join(
    DATA_DIR, "snp_features", "person_gene_all_chr.csv"
)

PHARMGKB_FILE = os.path.join(
    DATA_DIR, "curated", "pharmgkb_clean.csv"
)

OUT_FILE = os.path.join(
    DATA_DIR, "integrated", "person_drug.csv"
)

print("Gene matrix :", GENE_MATRIX_FILE)
print("PharmGKB    :", PHARMGKB_FILE)
print("Output      :", OUT_FILE)

# -------------------------------------------------
# Load gene → drug mapping from PharmGKB
# -------------------------------------------------
gene_to_drugs = defaultdict(set)

with open(PHARMGKB_FILE, "r") as f:
    reader = csv.DictReader(f)

    # adjust column names if needed
    gene_col = "gene"
    drug_col = "drug"

    for row in reader:
        gene = row[gene_col].strip()
        drug = row[drug_col].strip()

        if gene and drug:
            gene_to_drugs[gene].add(drug)

print(f"Loaded gene→drug links for {len(gene_to_drugs)} genes")

# -------------------------------------------------
# Read gene-level matrix and aggregate to drugs
# -------------------------------------------------
with open(GENE_MATRIX_FILE, "r") as f, open(OUT_FILE, "w", newline="") as out:
    reader = csv.reader(f)
    writer = csv.writer(out)

    header = next(reader)
    genes = header[1:]  # gene columns

    writer.writerow(["Person", "Drug", "Agg_Value"])

    for row in reader:
        person = row[0]
        drug_sum = defaultdict(int)

        for idx, gene in enumerate(genes, start=1):
            val = int(row[idx])
            if val == 0:
                continue

            for drug in gene_to_drugs.get(gene, []):
                drug_sum[drug] += val

        for drug, value in drug_sum.items():
            writer.writerow([person, drug, value])

print("DONE ✅ person_drug.csv created")
