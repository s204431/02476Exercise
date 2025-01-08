import matplotlib.pyplot as plt
import torch
import typer
import hydra
from pathlib import Path
import os

from exercise.model import MyAwesomeModel

from exercise.data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(hps, model_hps) -> None:
    """Train a model on MNIST."""
    path = Path(os.getcwd()).parent.parent.parent.absolute()
    print("Training day and night")
    print(f"{hps.lr=}, {hps.batch_size=}, {hps.epochs=}")

    model = MyAwesomeModel(model_hps).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=hps.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(hps.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), f"{path}/models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"{path}/reports/figures/training_statistics.png")

@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def main(cfg):
    train(cfg.training.hps, cfg.model.hps)

if __name__ == "__main__":
    main()
