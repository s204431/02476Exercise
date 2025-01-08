import torch
import typer
import hydra
from pathlib import Path
import os
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from exercise.model import MyAwesomeModel

from exercise.data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(hps) -> None:
    """Evaluate a trained model."""
    path = Path(os.getcwd()).parent.parent.parent.absolute()
    model_checkpoint = f"{path}/models/{hps.model_file}"
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = MyAwesomeModel(hps).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")

@hydra.main(config_name="config.yaml", config_path=f"{os.getcwd()}/configs")
def main(cfg):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
        evaluate(cfg.model.hps)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=30))
    #prof.export_chrome_trace("trace.json")


if __name__ == "__main__":
    main()
