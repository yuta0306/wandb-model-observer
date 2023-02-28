from typing import Iterable

import torch
import torch.nn as nn

import wandb


class ModelObserver:
    def __init__(self, model: nn.Module, key: str = "") -> None:
        self.key = key
        self.is_login = self.login(key=key)
        self.run = self.init()

        for n, mod in model.named_modules():
            if hasattr(mod, "weight"):
                self._register(module=mod, name=n, type_="weight")(mod.forward)
            if hasattr(mod, "bias"):
                self._register(module=mod, name=n, type_="bias")(mod.forward)
            self._register(module=mod, name=n, type_="output")(mod.forward)

    def _register(self, module: nn.Module, name: str, type_: str):
        def outer(func):
            def inner(*args, **kwargs):
                ret = func(*args, **kwargs)
                if type_ == "weight":
                    tensor = getattr(module, "weight")
                    tensor = torch.flatten(tensor).detach().cpu().numpy()
                    self.run.log(
                        {f"{name}.weight": wandb.Histogram(sequence=tensor)},
                        commit=False,
                    )
                if type_ == "bias":
                    tensor = getattr(module, "bias")
                    tensor = torch.flatten(tensor).detach().cpu().numpy()
                    self.run.log(
                        {f"{name}.bias": wandb.Histogram(sequence=tensor)}, commit=False
                    )
                if type_ == "output":
                    if isinstance(ret, torch.Tensor):
                        tensor = torch.flatten(ret).detach().cpu().numpy()
                        self.run.log(
                            {f"{name}.output": wandb.Histogram(sequence=tensor)},
                            commit=False,
                        )
                    elif isinstance(ret, dict):
                        for k, v in ret.items():
                            if isinstance(v, torch.Tensor):
                                tensor = torch.flatten(v).detach().cpu().numpy()
                                self.run.log(
                                    {
                                        f"{name}.{k}.output": wandb.Histogram(
                                            sequence=tensor
                                        )
                                    },
                                    commit=False,
                                )
                    elif isinstance(ret, Iterable):
                        for k, v in enumerate(ret):
                            if isinstance(v, torch.Tensor):
                                tensor = torch.flatten(v).detach().cpu().numpy()
                                self.run.log(
                                    {
                                        f"{name}.{k}.output": wandb.Histogram(
                                            sequence=tensor
                                        )
                                    },
                                    commit=False,
                                )
                return ret

            module.forward = inner
            return inner

        return outer

    def login(self, key: str) -> bool:
        return wandb.login(key=key)

    def init(self, *args, **kwargs):
        return wandb.init()

    def commit(self):
        self.run.log(data={}, commit=True)
