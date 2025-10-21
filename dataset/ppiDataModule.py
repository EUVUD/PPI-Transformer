import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

class ppiDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load your dataset here
        if stage == 'train':
            self.dataset = ppiDataset(f'{self.data_dir}/huri_train.csv')
        elif stage == 'val':
            self.dataset = ppiDataset(f'{self.data_dir}/huri_val.csv')
        elif stage == 'test':
            self.dataset = ppiDataset(f'{self.data_dir}/huri_test.csv')

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
