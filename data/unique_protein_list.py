import pandas as pd
import os

df = pd.read_csv("raw/HuRI/huri_train.csv")

# Collect all unique proteins
proteins = pd.concat([df[["Protein1_ID", "Protein1"]], df[["Protein2_ID", "Protein2"]].rename(columns={"Protein2_ID": "Protein1_ID", "Protein2": "Protein1"})])
unique_proteins = proteins.drop_duplicates(subset=["Protein1_ID"]).reset_index(drop=True)

# Rename columns for clarity
unique_proteins.columns = ["ProteinID", "Sequence"]
unique_proteins.to_csv("raw/HuRI/protein_sequences.csv", index=False)

print(f"✅ Found {len(unique_proteins)} unique proteins.")
