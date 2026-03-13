import csv
import os
import numpy as np

# -------------------------------------------------
# Project root (go up 3 levels from this file)
# src/labeling/create_adr_labels.py
# -------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR, "data", "final", "person_drug_side_effect.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR, "data", "final", "person_drug_side_effect_labeled.csv"
)

# -------------------------------------------------
# Load genetic scores
# -------------------------------------------------
scores = []

with open(INPUT_FILE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        scores.append(float(row["genetic_score"]))

# -------------------------------------------------
# Compute threshold (75th percentile)
# -------------------------------------------------
threshold = np.percentile(scores, 75)
print(f"[INFO] ADR threshold (75th percentile): {threshold}")

# -------------------------------------------------
# Write labeled dataset
# -------------------------------------------------
with open(INPUT_FILE, newline="", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames + ["ADR"]
    writer = csv.DictWriter(fout, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        score = float(row["genetic_score"])
        row["ADR"] = 1 if score >= threshold else 0
        writer.writerow(row)

print("[DONE] ADR-labeled dataset created")
print(f"Saved to: {OUTPUT_FILE}")
