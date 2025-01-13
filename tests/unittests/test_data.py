import os

import pytest
import torch
from torch.utils.data import Dataset

from exercise.data import corrupt_mnist


@pytest.mark.skipif(not os.path.exists("data/"), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist(testing=True)
    assert isinstance(train, Dataset), "Train dataset is not an instance of Dataset"
    assert isinstance(test, Dataset), "Test dataset is not an instance of Dataset"
    assert len(train) == 30000, "Train dataset did not have the correct number of samples"
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Input data does not have the correct shape"
            assert y in range(10), "Labels do not have the correct shape"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all(), "Not all labels are present in train data"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all(), "Not all labels are present in test data"
