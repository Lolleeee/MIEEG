from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
import torch
        

class AugmentationFunc(ABC):
    """Base class for data augmentation functions."""
    
    def __init__(self,):
        """
        Initialize the augmentation function.
        
        Args:
            requires_distribution_params: Whether this augmentation requires
                data distribution parameters to calibrate the magnitude.
        """
    
    def __call__(self, data: Dict[str, torch.Tensor], **kwargs) -> Any:
        """
        Apply the augmentation to the data.
        
        Args:
            data: Input data to augment.
            **kwargs: Additional parameters for augmentation.
        
        Returns:
            Augmented data.
        """

        model_inputs = data['input']
        target = data['target']
        augmented_input, augmented_target = self.compute_augmentation(model_inputs, target)

        return {'input': augmented_input, 'target': augmented_target}
    
    @abstractmethod
    def compute_augmentation(self, model_inputs: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute and apply the augmentation.
        
        Args:
            data: Input data to augment.
            **kwargs: Additional parameters for augmentation.
        
        Returns:
            Augmented data.
        """
        pass

class AddGaussianNoise(AugmentationFunc):
    """Add Gaussian noise to the input data."""
    
    def __init__(self, noise_std: float = 0.01):
        """
        Initialize the Gaussian noise augmentation.
        
        Args:
            noise_std: Standard deviation of the Gaussian noise.
        """
        super().__init__()
        self.noise_std = noise_std
    
    def compute_augmentation(self, model_inputs: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add Gaussian noise to the model inputs.
        
        Args:
            model_inputs: Input data to augment.
            target: Target data (not modified).
        
        Returns:
            Tuple of augmented model inputs and original target.
        """
        noise = torch.randn_like(model_inputs) * self.noise_std
        augmented_inputs = model_inputs + noise
        return augmented_inputs, target