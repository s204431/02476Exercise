import hydra
import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, hps) -> None:
        super().__init__()
        self.hps = hps
        self.conv1 = nn.Conv2d(1, hps.hidden1, hps.kernel_size, hps.stride)
        self.conv2 = nn.Conv2d(hps.hidden1, hps.hidden2, hps.kernel_size, hps.stride)
        self.conv3 = nn.Conv2d(hps.hidden2, hps.hidden3, hps.kernel_size, hps.stride)
        self.dropout = nn.Dropout(hps.dropout)
        self.fc1 = nn.Linear(hps.hidden3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, self.hps.pool_size, self.hps.pool_stride)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, self.hps.pool_size, self.hps.pool_stride)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, self.hps.pool_size, self.hps.pool_stride)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
