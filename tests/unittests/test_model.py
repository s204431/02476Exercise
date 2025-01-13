import os

import pytest
import torch
from hydra import compose, initialize

from exercise.model import MyAwesomeModel


def test_model():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    model = MyAwesomeModel(cfg.model.hps)
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10), "Output shape is not correct"


def test_error_on_wrong_shape():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    model = MyAwesomeModel(cfg.model.hps)
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    with pytest.raises(ValueError, match="Expected each sample to have shape \[1, 28, 28\]"):
        model(torch.randn(1, 1, 28, 29))


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int) -> None:
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    model = MyAwesomeModel(cfg.model.hps)
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)
