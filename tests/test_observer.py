import torch
import torch.nn as nn

from wandb_model_observer import ModelObserver


class Module(nn.Module):
    def __init__(self) -> None:
        super(Module, self).__init__()
        self.linear = nn.Linear(128, 128)

    def forward(self, inputs):
        return self.linear(inputs)


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.linear = nn.Linear(128, 2)
        self.mod1 = Module()

    def forward(self, inputs):
        out = self.mod1(inputs)
        out = self.linear(out)
        return out


def test_observer():
    model = Model()
    observer = ModelObserver(model, "")
    for _ in range(50):
        _ = model(torch.randn(2, 128))
    observer.commit()
