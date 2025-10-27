from ppiDataModule import ppiDataModule

dm = ppiDataModule(data_dir="../data", batch_size=32)
dm.setup()

train_loader = dm.train_dataloader()
batch = next(iter(train_loader))
emb1, emb2, label = batch
print(emb1.shape, emb2.shape, label.shape)
