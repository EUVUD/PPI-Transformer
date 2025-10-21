import pandas as pd
import torch
from transformers import EsmTokenizer
from transformers import EsmModel

def generate_esm_tokens():
    model = 'facebook/esm2_t6_8M_UR50D'
    tokenizer = EsmTokenizer.from_pretrained(model)
    esm_model = EsmModel.from_pretrained(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device)

    data = pd.read_csv('../data/huri_unique_proteins.csv')

    for row in data.itertuples(index=False):
        proteinID = row.Protein_ID
        sequence = row.Protein
        inputs = tokenizer(sequence, return_tensors='pt', 
                           truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = esm_model(**inputs)
        embeddings = outputs.last_hidden_state
        torch.save(embeddings, f'../data/esm_embeddings/{proteinID}.pt')

if __name__ == "__main__":
    generate_esm_tokens()
        


