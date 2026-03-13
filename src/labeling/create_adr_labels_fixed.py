import pandas as pd
import numpy as np

df = pd.read_csv("data/final/person_drug_side_effect.csv")

np.random.seed(42)

def generate_labels(group):

    n = len(group)

    # 20% positive ADR
    k = max(1, int(0.2 * n))

    labels = np.zeros(n)
    labels[:k] = 1
    np.random.shuffle(labels)

    group["adr_label"] = labels
    return group

df = df.groupby(["person_id","drug_name"]).apply(generate_labels)

df.reset_index(drop=True, inplace=True)

df.to_csv("data/final/person_drug_side_effect_adr_labeled.csv", index=False)

print("ADR labels generated")
print(df["adr_label"].value_counts())