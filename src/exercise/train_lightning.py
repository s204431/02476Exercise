from pytorch_lightning import Trainer, loggers
from exercise.model_lightning import MyAwesomeModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import hydra
import os
from exercise.data import corrupt_mnist
import torch

def train(hps, hps_model):
    train_set, test_set = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hps.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=hps.batch_size)
    model = MyAwesomeModel(hps_model, hps.lr)  # this is our LightningModule
    early_stopping_callback = EarlyStopping(
    monitor="val_loss", patience=3, verbose=True, mode="min"
    )
    trainer = Trainer(max_epochs=hps.epochs, limit_train_batches=0.2, callbacks=[early_stopping_callback], logger=loggers.WandbLogger(project="wandb_test"))  # this is our Trainer
    trainer.fit(model, train_dataloader, test_dataloader)

@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def main(cfg):
    train(cfg.training.hps, cfg.model.hps)

if __name__ == "__main__":
    main()
