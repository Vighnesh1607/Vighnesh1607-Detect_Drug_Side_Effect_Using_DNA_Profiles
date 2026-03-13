import csv
import os
from collections import defaultdict

# ---------------- Paths ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "snp_features")

GENE_FILE = os.path.join(DATA_DIR, "genes_chr6_7_10_12_16.csv")
OUT_FILE  = os.path.join(DATA_DIR, "person_gene_all_chr.csv")

CHR_FILES = {
    "chr6": "chr6_maf_filtered.csv",
    "chr7": "chr7_maf_filtered.csv",
    "chr10": "chr10_maf_filtered.csv",
    "chr12": "chr12_maf_filtered.csv",
    "chr16": "chr16_maf_filtered.csv",
}

# ---------------- Load gene ranges ----------------
genes_by_chr = defaultdict(list)

with open(GENE_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        chrom = row["chromosome"]
        genes_by_chr[chrom].append(
            (row["gene"], int(row["start"]), int(row["end"]))
        )

for chrom in genes_by_chr:
    genes_by_chr[chrom].sort(key=lambda x: x[1])

def find_gene(chrom, pos):
    for gene, start, end in genes_by_chr[chrom]:
        if start <= pos <= end:
            return gene
        if pos < start:
            break
    return None

# ---------------- Prepare global containers ----------------
all_genes = []
person_gene = defaultdict(lambda: defaultdict(int))

# ---------------- Process each chromosome ----------------
for chrom, file_name in CHR_FILES.items():
    snp_file = os.path.join(DATA_DIR, file_name)

    print("Processing", chrom)

    with open(snp_file) as f:
        reader = csv.reader(f)
        header = next(reader)

        col_to_gene = {}
        for idx, col in enumerate(header):
            if col == "Person":
                continue
            try:
                pos = int(col.split("_")[-1])
            except:
                continue

            gene = find_gene(chrom, pos)
            if gene:
                col_to_gene[idx] = gene
                if gene not in all_genes:
                    all_genes.append(gene)

        for row in reader:
            person = row[0]
            for idx, gene in col_to_gene.items():
                person_gene[person][gene] += int(row[idx])

# ---------------- Write final WIDE matrix ----------------
with open(OUT_FILE, "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["Person"] + all_genes)

    for person in sorted(person_gene, key=int):
        writer.writerow(
            [person] + [person_gene[person].get(g, 0) for g in all_genes]
        )

print("DONE ✅ person_gene_all_chr.csv created")
