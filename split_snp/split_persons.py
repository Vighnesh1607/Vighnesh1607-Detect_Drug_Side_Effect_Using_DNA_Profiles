import pandas as pd
import os

# Input file in same folder as script
input_file = "person_gene_all_chr.csv"

# Output folder
output_folder = "person_files"
os.makedirs(output_folder, exist_ok=True)

# Read CSV
df = pd.read_csv(input_file)

print(f"Total rows: {len(df)}")
print(f"Persons found: {df['Person'].nunique()}")

# Create one file per person
for person_id in df["Person"].unique():
    person_df = df[df["Person"] == person_id]

    output_file = os.path.join(
        output_folder,
        f"person_{person_id}.csv"
    )

    person_df.to_csv(output_file, index=False)

    print(f"Created: {output_file}")

print("All files created successfully.")