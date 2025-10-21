from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class ppiDataset(Dataset):
    def __init__(self, data_path):
        # Load your data from the file
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        # Example: Load data from a CSV file
        df = pd.read_csv(data_path)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single data point (e.g., features and labels)
        protein1_ID = self.data.iloc[idx]['Protein1_ID']
        protein2_ID = self.data.iloc[idx]['Protein2_ID']
        label = self.data.iloc[idx]['Y']

        embedding1 = torch.load(f'../data/esm_embeddings/{protein1_ID}.pt')
        embedding2 = torch.load(f'../data/esm_embeddings/{protein2_ID}.pt')

        return embedding1, embedding2, label