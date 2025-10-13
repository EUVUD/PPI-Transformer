from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
from tqdm import tqdm

def generate_esm_embeddings(sequence_file, output_dir, model_name="facebook/esm2_t6_8M_UR50D"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(sequence_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            seq_id, seq = row["ProteinID"], row["Sequence"]
            tokens = tokenizer(seq, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**tokens)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
            torch.save(embedding, os.path.join(output_dir, f"{seq_id}.pt"))

if __name__ == "__main__":
    generate_esm_embeddings("raw/HuRI/protein_sequences.csv", "raw/HuRI/embeddings/")
