import os
import random
import logging
import numpy as np
import torch

from packages.train.trainer_config_schema import TrainerConfig

logging.basicConfig(level=logging.INFO)

def _set_seed(config: TrainerConfig = None, seed: int = None):
        """
        Set random seeds for reproducibility across:
        - Python's random module
        - NumPy
        - PyTorch (CPU and GPU)
        - CuDNN (for CUDA operations)
        """

        if config is not None and seed is None:
            assert isinstance(config, TrainerConfig), "config must be an instance of TrainerConfig"
            seed = config.seed if seed is None else seed
        elif config is not None and seed is not None:
            logging.warning("Both config and seed are provided. The seed from config will be overridden by the provided seed.")
        

        if seed is not None:
            logging.info(f"Setting random seed to {seed} for reproducibility, note that this may slow down training.")
            
            # Python random
            random.seed(seed)
            
            # NumPy
            np.random.seed(seed)
            
            # PyTorch CPU
            torch.manual_seed(seed)
            
            # PyTorch GPU (if available)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU
            
            # CuDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set PYTHONHASHSEED for complete reproducibility
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            logging.info("Random seed set successfully")
        else:
            logging.info("No seed specified - training will not be deterministic")