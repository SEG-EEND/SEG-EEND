#!/usr/bin/env python3

# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

import torch.optim as optim
from torch.nn import Module
from types import SimpleNamespace
from typing import Any, Dict
import torch.distributed as dist

class NoamOpt:
    """ Optim wrapper that implements rate scheduling. """
    def __init__(self, model_size: int, warmup: int, optimizer: optim, scale_factor: float = 1.0) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0
        self.scale_factor = scale_factor  # Apply the scaling factor

    def state_dict(self) -> Dict[str, Any]:
        """ Returns the state of the warmup scheduler as a :class:`dict`."""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Loads the warmup scheduler's state."""
        self.__dict__.update(state_dict)

    def step(self) -> None:
        """ Update parameters and rate """
        self._step += 1
        self._rate = self.rate(self._step)  # Update learning rate for the current step
        for p in self.optimizer.param_groups:
            p['lr'] = self._rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        """ Compute learning rate """
        step = max(1, step if step is not None else self._step)  # Ensure step is at least 1
        warmup_factor = min(step ** (-0.5), step * self.warmup ** (-1.5))  # Apply Noam formula
        return self.scale_factor * (self.model_size ** (-0.5)) * warmup_factor  #  Scale factor 적용

    def get_rate(self) -> float:
        return self._rate if self._rate is not None and self._rate > 0 else self.rate(self._step)  # Compute rate if _rate is zero

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


def setup_optimizer(args: SimpleNamespace, model: Module) -> optim:
    """ Setup optimizer for the model. """
    world_size = dist.get_world_size() if dist.is_initialized() else 1  # Check the number of GPUs
    scale_factor = args.lr / 0.001  # Adjust scaling factor precisely

    print(f" [DEBUG] Optimizer type: {args.optimizer}, Scale Factor: {scale_factor}, World Size: {world_size}")

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'noam':
        optimizer = NoamOpt(
            model_size=args.hidden_size,  
            warmup=args.noam_warmup_steps,
            optimizer=optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9),
            scale_factor=scale_factor,  # Adjust learning rate precisely
        )
    else:
        raise ValueError(f" Unknown optimizer type: {args.optimizer}")

    print(f" [DEBUG] Optimizer initialized: {optimizer}")  #  Debugging information
    return optimizer

def get_rate(optimizer) -> float:
    """ Retrieve the learning rate from NoamOpt or standard optimizers. """
    if isinstance(optimizer, NoamOpt):
        rate = optimizer.get_rate()
        print(f" [DEBUG] NoamOpt learning rate: {rate}")  #  Debugging information
    else:
        if len(optimizer.param_groups) == 0:
            raise ValueError(" Optimizer has no parameter groups! Check optimizer initialization.")  #  Debugging information

        rate = optimizer.param_groups[0]['lr']
        print(f" [DEBUG] Standard optimizer learning rate: {rate}")  #  Debugging information

    if rate is None:
        raise ValueError(" get_rate() returned None! Check optimizer initialization.")  #  Debugging information

    return rate