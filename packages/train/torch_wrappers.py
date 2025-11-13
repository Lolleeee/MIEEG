import logging
import torch
import torch.nn as nn
from typing import Dict, Union, Tuple, Any, List
from abc import abstractmethod

class TorchWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_io_signature(outputs: Dict[str, torch.Tensor], inputs: torch.Tensor):
        """
        Returns a hashable signature (shapes, keys, dtypes) for detecting structural changes.
        """
        output_sig = tuple(sorted((k, tuple(v.shape), v.dtype) for k, v in outputs.items()))
        input_sig = (tuple(inputs['input'].shape), inputs['input'].dtype, tuple(inputs['target'].shape), inputs['target'].dtype)
        return (output_sig, input_sig)
    
    def _validate_inputs(self, inputs: Dict) -> torch.Tensor:
        """
        Validate input targets.
        
        Args:
            inputs: Model inputs which must contain ground truth targets
            outputs: Normalized outputs dict (for shape checking)
            
        Returns:
            inputs: Validated inputs
        """
        if not isinstance(inputs, dict):
            raise TypeError(f"Inputs must be a dict, got {type(inputs)}")

        # Check for NaN/Inf
        if torch.isnan(inputs['input']).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(inputs['input']).any():
            raise ValueError("Input contains Inf values")
        if torch.isnan(inputs['target']).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(inputs['target']).any():
            raise ValueError("Input contains Inf values")
            
        return inputs
    
    def _validate_and_normalize_outputs(self, outputs: Union[torch.Tensor, Dict, Tuple]) -> Dict[str, torch.Tensor]:
        # Convert to dict (same logic as before)
        if isinstance(outputs, torch.Tensor):
            outputs = {'main_output': outputs}
        elif isinstance(outputs, tuple):
            outputs = {f'main_output' if i == 0 else f'output_{i}': out for i, out in enumerate(outputs)}
        elif isinstance(outputs, dict):
            for key, value in outputs.items():
                if not isinstance(value, (torch.Tensor, int, float)):
                    raise TypeError(f"Output '{key}' must be a tensor or scalar, got {type(value)}")
        else:
            raise TypeError(f"Outputs must be Tensor, Dict, or Tuple. Got {type(outputs)}")

        if self.expected_output_keys is not None:
            missing = self.expected_output_keys - set(outputs.keys())
            if missing:
                logging.warning(
                    f"[{self.name}] Model Output keys mismatch. "
                    f"Missing keys: {missing if missing else 'None'}, "
                    f"Check the {self.name} __init__."
                )

        return outputs
    
    def disable_validation(self):
        """Disable validation for performance (use in production)"""
        self._validation_enabled = False

    def enable_validation(self):
        """Enable validation (default, use during development/debugging)"""
        self._validation_enabled = True
    
class TorchMetric(TorchWrapper):
    """
    Base class for all metrics with automatic input/output validation.

    To create a custom metric, subclass and implement `_compute_metric()`:


    Example:
        class MyMetric(TorchMetric):
            def _compute_metric(self, outputs: Dict, inputs: torch.Tensor) -> torch.Tensor:
                reconstruction = outputs['reconstruction']
                return F.mse_loss(reconstruction, inputs)
    """
    def __init__(self, expected_model_output_keys: Union[List[str], None] = None):
        super().__init__()
        self.name: str = self.__class__.__name__
        self._validation_enabled = True
        self._last_io_signature = None
        self._warning_emitted = False

        self.expected_output_keys = set(expected_model_output_keys) if expected_model_output_keys else None

    def __call__(self, outputs: Union[torch.Tensor, Dict, Tuple], inputs: torch.Tensor) -> torch.Tensor:
        if self._validation_enabled:
            val_outputs = self._validate_and_normalize_outputs(outputs)
            val_inputs = self._validate_inputs(inputs)
            
            if not self._warning_emitted:
                val_signature = self._get_io_signature(val_outputs, val_inputs)
                raw_signature = self._get_io_signature(outputs, inputs)
                if val_signature != raw_signature:
                    logging.warning(
                        f"[{self.name}] Detected change in input/output structure.\n"
                        f"Pre-Validation signature: {raw_signature}\n"
                        f"Validated signature: {val_signature}\n"
                        f"Make sure to fix this if validation is disabled."
                    )
                    self._warning_emitted = True

            metric = self._compute_metric(val_outputs, val_inputs)
        else:
            metric = self._compute_metric(outputs, inputs)

        metric = self._validate_metric_output(metric)

        return metric

    @abstractmethod
    def _compute_metric(self, outputs: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        """
        Implement this method in subclasses to compute the actual metric.

        Args:
            outputs: Normalized dict of model outputs
            inputs: Validated ground truth targets
            
        Returns:
            metric: Computed metric (scalar tensor)
        """
        raise NotImplementedError("Subclasses must implement _compute_metric()")

    def _validate_metric_output(self, metric: torch.Tensor) -> Dict:
        """
        Validate the computed metric value.

        Args:
            metric: Computed metric
            
        Returns:
            metric: Validated metric
        """

        # Check for NaN/Inf
        if torch.isnan(metric).any():
            raise ValueError(f"Metric '{metric}' is NaN")
        if torch.isinf(metric).any():
            raise ValueError(f"Metric '{metric}' is Inf")
        if metric < 0:
            raise ValueError(f"Metric '{metric}' is negative: {metric}")

        return metric

class TorchLoss(TorchWrapper):
    """
    Base class for all loss functions with automatic input/output validation.
    
    To create a custom loss, subclass and implement `_compute_loss()`:
    
    Example:
        class MyLoss(TorchLoss):
            def _compute_loss(self, outputs: Dict, inputs: torch.Tensor) -> torch.Tensor:
                reconstruction = outputs['reconstruction']
                return F.mse_loss(reconstruction, inputs)
    """
    def __init__(self, expected_model_output_keys: Union[List[str], None] = None, expected_loss_keys: Union[List[str], None] = None):
        super().__init__()
        self.name: str = self.__class__.__name__
        self._validation_enabled = True
        self._last_io_signature = None
        self._warning_emitted = False

        self.expected_output_keys = set(expected_model_output_keys) if expected_model_output_keys else None
        self.expected_loss_keys = set(expected_loss_keys) if expected_loss_keys else 'loss'

    def forward(self, outputs: Union[torch.Tensor, Dict, Tuple], inputs: torch.Tensor) -> torch.Tensor:
        if self._validation_enabled:
            val_outputs = self._validate_and_normalize_outputs(outputs)
            val_inputs = self._validate_inputs(inputs)
            # --- Track I/O signature ---
            if not self._warning_emitted:
                val_signature = self._get_io_signature(val_outputs, val_inputs)
                raw_signature = self._get_io_signature(outputs, inputs)
                if val_signature != raw_signature:
                    logging.warning(
                        f"[{self.name}] Detected change in input/output structure.\n"
                        f"Pre-Validation signature: {raw_signature}\n"
                        f"Validated signature: {val_signature}\n"
                        f"Make sure to fix this if validation is disabled."
                    )
                    self._warning_emitted = True

            loss = self._compute_loss(val_outputs, val_inputs)
        else:
            loss = self._compute_loss(outputs, inputs)

        loss = self._validate_loss_output(loss)

        return loss

    @abstractmethod
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        """
        Implement this method in subclasses to compute the actual loss.
        
        Args:
            outputs: Normalized dict of model outputs
            inputs: Validated ground truth targets
            
        Returns:
            loss: Computed loss (scalar tensor)
        """
        raise NotImplementedError("Subclasses must implement _compute_loss()")

    def _validate_loss_output(self, loss: Dict) -> Dict:
        """
        Validate the computed loss value.

        Args:
            loss: Computed loss
            
        Returns:
            loss: Validated loss
        """

        if not isinstance(loss, Dict):
            raise TypeError(f"Loss output must be a dict, got {type(loss)}")
        if 'loss' not in loss:
            raise KeyError("Loss output dict must contain 'loss' key for main loss value")

        if self.expected_loss_keys is not None:
            missing = self.expected_loss_keys - set(loss.keys())
            if missing:
                logging.warning(
                    f"[{self.name}] loss output keys mismatch. "
                    f"Missing keys: {missing if missing else 'None'}, "
                    f"Check the loss class __init__."
                )
        return loss 