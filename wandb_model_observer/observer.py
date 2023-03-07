from typing import Iterable

import torch
import torch.nn as nn

import wandb


class ModelObserver:
    def __init__(
        self,
        model: nn.Module,
        key: str = "",
        *args,
        **kwargs,
    ) -> None:
        self.key = key
        self.is_login = self.login(key=key)
        self.run = self.init(*args, **kwargs)

        for n, mod in model.named_modules():
            self._register_weight(name=n, module=mod)
            self._register_output(name=n, module=mod)
        for n, param in model.named_parameters():
            self._register_gradient(name=n, param=param)

    def _register_gradient(
        self,
        name: str,
        param: "torch.nn.Parameter",
        graph_type: str = "distribution",
    ) -> None:
        if not isinstance(param, torch.autograd.Variable):
            return

        if not param.requires_grad:
            return

        def log(grad: torch.Tensor):
            if graph_type == "distribution":
                tensor = torch.flatten(grad).detach().cpu().numpy()
                self.run.log(
                    {f"gradients/{name}": wandb.Histogram(sequence=tensor)},
                    commit=False,
                )
            else:
                raise NotImplementedError

        param.register_hook(log)

    def _register_weight(
        self,
        name: str,
        module: "torch.nn.Module",
        graph_type: str = "distribution",
    ) -> None:
        monitor = []
        if hasattr(module, "weight"):
            monitor.append("weight")
        if hasattr(module, "bias"):
            monitor.append("bias")

        def log(module, grad_in, grad_out, monitor=monitor):
            for m in monitor:
                if graph_type == "distribution":
                    tensor = getattr(module, m)
                    tensor = torch.flatten(tensor).detach().cpu().numpy()
                    self.run.log(
                        {f"parameters/{name}.{m}": wandb.Histogram(sequence=tensor)},
                        commit=False,
                    )
                else:
                    raise NotImplementedError

        if len(monitor) > 0:
            module.register_full_backward_hook(log)

    def _register_output(
        self,
        name: str,
        module: "torch.nn.Module",
        graph_type: str = "distribution",
    ) -> None:
        def log(module, input, output):
            outputs = {}
            if isinstance(output, torch.Tensor):
                outputs.update({name: output})
            elif isinstance(output, dict):
                outputs.update(output)
            elif isinstance(output, Iterable):
                outputs.update({f"{name}.{i}": output[i] for i in range(len(output))})

            for k, v in outputs.items():
                if graph_type == "distribution":
                    tensor = torch.flatten(v).detach().cpu().numpy()
                    self.run.log(
                        {f"outputs/{k}": wandb.Histogram(sequence=tensor)},
                        commit=False,
                    )
                else:
                    raise NotImplementedError

        module.register_forward_hook(log)

    def login(self, key: str) -> bool:
        return wandb.login(key=key)

    def init(self, *args, **kwargs):
        return wandb.init(*args, **kwargs)

    def commit(self):
        self.run.log(data={}, commit=True)
