import hydra
import torch
from pytorch_lightning import LightningModule
from torch import nn


class MyAwesomeModel(LightningModule):
    """My awesome model."""

    def __init__(self, hps, lr) -> None:
        super().__init__()
        self.hps = hps
        self.lr = lr
        self.conv1 = nn.Conv2d(1, hps.hidden1, hps.kernel_size, hps.stride)
        self.conv2 = nn.Conv2d(hps.hidden1, hps.hidden2, hps.kernel_size, hps.stride)
        self.conv3 = nn.Conv2d(hps.hidden2, hps.hidden3, hps.kernel_size, hps.stride)
        self.dropout = nn.Dropout(hps.dropout)
        self.fc1 = nn.Linear(hps.hidden3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, self.hps.pool_size, self.hps.pool_stride)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, self.hps.pool_size, self.hps.pool_stride)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, self.hps.pool_size, self.hps.pool_stride)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch, batch_idx):
        img, target = batch
        y_pred = self(img)
        loss = nn.CrossEntropyLoss()(y_pred, target)
        acc = (target == y_pred.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = nn.CrossEntropyLoss()(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
