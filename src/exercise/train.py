import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import typer
import wandb
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

from exercise.data import corrupt_mnist
from exercise.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(hps, model_hps) -> None:
    """Train a model on MNIST."""
    path = Path(os.getcwd()).parent.parent.parent.absolute()
    print("Training day and night")
    lr = hps.lr
    batch_size = hps.batch_size
    epochs = hps.epochs
    print(f"{lr=}, {batch_size=}, {epochs=}")
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": hps.lr, "batch_size": hps.batch_size, "epochs": hps.epochs},
    )

    model = MyAwesomeModel(model_hps).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads)})

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot,
                preds[:, class_id],
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )

        plt_image = wandb.Image(plt.gcf())
        plt.clf()
        wandb.log({"roc_curve": plt_image})
        # wandb.log({"roc": plt})
        # alternatively the wandb.plot.roc_curve function can be used
    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    torch.save(model.state_dict(), f"{path}/models/model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file(f"{path}/models/model.pth")
    run.log_artifact(artifact)


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def main(cfg):
    train(cfg.training.hps, cfg.model.hps)


if __name__ == "__main__":
    main()
