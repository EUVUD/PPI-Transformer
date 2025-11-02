import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.ppiDataset import ppiDataset
from dataset.collate_fn import collate_fn

class ppiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load your dataset here
        self.train_dataset = ppiDataset(f'{self.data_dir}/huri_neg_train.csv')
        self.val_dataset = ppiDataset(f'{self.data_dir}/huri_neg_val.csv')
        self.test_dataset = ppiDataset(f'{self.data_dir}/huri_neg_test.csv')

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)
        print(f"train_dataloader created with id={id(loader)}")
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        print(f"val_dataloader created with id={id(loader)}")
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        print(f"test_dataloader created with id={id(loader)}")
        return loader
