from pathlib import Path

from pytorch_lightning import Trainer
from model.ppiPredictor import ppiPredictor
from dataset.ppiDataModule import ppiDataModule
import wandb
from pytorch_lightning.loggers import WandbLogger

def train_model():
    wandb_logger = WandbLogger(project="PPI-Transformer")

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    data_module = ppiDataModule(data_dir=str(data_dir), batch_size=32)
    model = ppiPredictor(lr=1e-4)

    wandb_logger.experiment.config.update({
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 1
    })

    trainer = Trainer(max_epochs=1, logger=wandb_logger, accelerator="gpu", devices=1)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    wandb.finish()

if __name__ == "__main__":
    train_model()