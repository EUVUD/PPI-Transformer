import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ppiDataset import ppiDataset
from collate_fn import collate_fn

class ppiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load your dataset here
        self.train_dataset = ppiDataset(f'{self.data_dir}/huri_train.csv')
        self.val_dataset = ppiDataset(f'{self.data_dir}/huri_val.csv')
        self.test_dataset = ppiDataset(f'{self.data_dir}/huri_test.csv')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
