import torch
import torch.nn as nn
from typing import Dict, Union, List, Tuple
import logging
from abc import abstractmethod

from packages.train.torch_wrappers import TorchMetric
    
class MSE(TorchMetric):
    def __init__(self):
        super().__init__(expected_model_output_keys=['reconstruction'])
        self.name = "MSE"
        self.func = torch.nn.functional.mse_loss

    def __call__(self, outputs, targets) -> torch.Tensor:
        rec = outputs['reconstruction']
        return self.func(rec, targets)

class MAE(TorchMetric):
    def __init__(self):
        super().__init__(expected_model_output_keys=['reconstruction'])
        self.name = "MAE"
        self.func = torch.nn.functional.l1_loss

    def __call__(self, outputs, targets) -> torch.Tensor:
        rec = outputs['reconstruction']
        return self.func(rec, targets)

class RMSE(TorchMetric):
    def __init__(self):
        super().__init__()
        self.name = "RMSE"

    def __call__(self, outputs, targets) -> torch.Tensor:
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(targets, tuple):
            targets = targets[0]
        return torch.sqrt(torch.mean((outputs - targets) ** 2))

class AxisCorrelation(TorchMetric):
    def __init__(self, axis=0):
        super().__init__()
        self.name = f"AxisCorrelation_axis{axis}"
        self.axis = axis

    def __call__(self, outputs, targets) -> torch.Tensor:
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(targets, tuple):
            targets = targets[0]
        outputs_mean = torch.mean(outputs, dim=self.axis, keepdim=True)
        targets_mean = torch.mean(targets, dim=self.axis, keepdim=True)

        numerator = torch.sum((outputs - outputs_mean) * (targets - targets_mean), dim=self.axis)
        denominator = torch.sqrt(torch.sum((outputs - outputs_mean) ** 2, dim=self.axis) * 
                                 torch.sum((targets - targets_mean) ** 2, dim=self.axis))

        correlation = numerator / (denominator + 1e-8)  # Avoid division by zero
        return torch.mean(correlation)  # Return mean correlation across other axes