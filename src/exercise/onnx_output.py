import torch
import torchvision
from hydra import compose, initialize
from model import MyAwesomeModel
from pathlib import Path
import os

with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="config")

model = MyAwesomeModel(cfg.model.hps)
path = Path(os.getcwd()).absolute()
model_checkpoint = f"{path}/models/{cfg.model.hps.model_file}"
model.load_state_dict(torch.load(model_checkpoint))
model.eval()
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model=model,
    args=(dummy_input,),
    f="mymodel.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)