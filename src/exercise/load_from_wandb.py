import os

import hydra
import torch
import wandb

from exercise.model import MyAwesomeModel


@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def main(cfg):
    run = wandb.init()
    artifact = run.use_artifact(
        "s204431-technical-university-of-denmark-org/wandb-registry-model/Test:latest", type="model"
    )
    artifact_dir = artifact.download("artifacts")
    model = MyAwesomeModel(cfg.model.hps)
    model.load_state_dict(torch.load("artifacts/model.pth"))


if __name__ == "__main__":
    main()
