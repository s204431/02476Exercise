import os
import time

import torch
import wandb
from hydra import compose, initialize

from exercise.model import MyAwesomeModel, load_from_checkpoint


def load_model(artifact, hps):
    logdir = "artifacts"
    model_checkpoint="s204431-technical-university-of-denmark-org/wandb-registry-model/Test:latest"
    wandb.login(key=[os.getenv("WANDB_API_KEY")])
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(model_checkpoint)
    artifact.download(root=logdir)
    file_name = artifact.files()[0].name
    return load_from_checkpoint(f"{logdir}/{file_name}", hps)

def test_model_speed():
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")
    model = load_model(os.getenv("MODEL_NAME"), cfg.model.hps)
    start = time.time()
    for _ in range(100):
        model(torch.rand(1, 1, 28, 28))
    end = time.time()
    assert end - start < 1