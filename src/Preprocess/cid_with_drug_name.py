import pandas as pd
import requests
import time
import os

INPUT_FILE = "distinct_cids.csv"

# ⚠️ CHANGE OUTPUT PATH (NOT OneDrive)
OUTPUT_DIR = r"C:\pubchem_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cid_with_drug_names_final.csv")

DELAY = 0.2

# Create folder if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df["cid_numeric"] = df["drug_id"].str.replace("CID", "", regex=False).astype(int)

results = []

for i, row in df.iterrows():
    cid = row["cid_numeric"]
    drug_id = row["drug_id"]

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title/JSON"

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            name = r.json()["PropertyTable"]["Properties"][0]["Title"]
        else:
            name = "Not Found"
    except:
        name = "Error"

    results.append([drug_id, name])
    print(f"✅ {i+1}/{len(df)} | {drug_id} → {name}")

    time.sleep(DELAY)

# Save ONCE at the end
out_df = pd.DataFrame(results, columns=["drug_id", "drug_name"])
out_df.to_csv(OUTPUT_FILE, index=False)

print("\n🎉 SUCCESS!")
print("📄 File saved at:", OUTPUT_FILE)
