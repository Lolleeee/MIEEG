import logging
import torch
from typing import Dict, TYPE_CHECKING, Tuple
from packages.train.trainer_config_schema import PlotType, SanityCheckConfig, TrainerConfig, DatasetType

from packages.train.training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    force=True
)
#TODO actually check that with a non random dataset the model overfits the training set lowering the loss
#TODO setup the test loader class in the config building, maybe also a option to decide the dataset class for sanity check
class SanityChecker(Trainer):
    """
    A sanity checker that runs a quick training session with a subset of data
    to verify the training pipeline works correctly before full training.
    
    Inherits from Trainer to automatically stay in sync with any Trainer modifications.
    Main features is that it should turn on all validation in all objects
    and check that the model overfits the training set lowering the loss.
    After the sanity check validation can and should be turned off again to save overhead.
    """

    def __init__(self, config: TrainerConfig):
        """
        Args:
            config: Original trainer configuration
            sanity_config: Optional dict to override specific sanity check parameters:
                - train_split: float, default 0.2
                - val_split: float, default 0.2
                - test_split: float, default 0.6
                - epochs: int, default 2
                - batch_size: int, optional (uses original if not set)
        """

        self.trainer_config = config
        assert isinstance(self.trainer_config, TrainerConfig)
        

        self.sanity_config = self.trainer_config.sanity_check
        assert isinstance(self.sanity_config, SanityCheckConfig)
        self.sanity_config = self.sanity_config.model_dump()

        self.sanity_checked_trainer_config, self.single_epoch_sanity_checked_trainer_config = self._prepare_sanity_config()

        logging.info("=" * 60)
        logging.info("SANITY CHECK MODE ACTIVATED")
        logging.info("=" * 60)

    def _prepare_sanity_config(self) -> Tuple[TrainerConfig, TrainerConfig]:
        """
        Modify the configuration for sanity checking.
        
        Args:
            config: Original configuration dict
            
        Returns:
            Modified configuration dict for sanity checking
        """
        sanity_checked_trainer_config = self.trainer_config.model_dump()
        # Remove sanity check config to avoid recursion
        sanity_checked_trainer_config['sanity_check'] = None
        # Set epochs and data splits
        sanity_checked_trainer_config['train_loop']['epochs'] = self.sanity_config['epochs']
        sanity_checked_trainer_config['dataset']['data_loader']['set_sizes'] = self.sanity_config['set_sizes']

        # Remove all helpers like early stopping and such to avoid interference
        for helper in sanity_checked_trainer_config["helpers"]:
            if helper == 'history_plot':
                sanity_checked_trainer_config["helpers"][helper]['plot_type'] = PlotType.OFF
                sanity_checked_trainer_config["helpers"][helper]['save_path'] = None
            else:
                sanity_checked_trainer_config["helpers"][helper] = None

        # Remove all metrics to avoid interference
        sanity_checked_trainer_config["info"]["metrics"] = []
        sanity_checked_trainer_config["info"]["metrics_args"] = None

        # Remove all other non-essential configs that may interfere
        sanity_checked_trainer_config['gradient_control']['grad_clip'] = None
        sanity_checked_trainer_config['optimizer']['weight_decay'] = 0.0
        sanity_checked_trainer_config['optimizer']['asym_lr'] = None
        sanity_checked_trainer_config['gradient_control']['use_amp'] = False

        # Set the dataset to a TestTorchDataset to subsample
        sanity_checked_trainer_config['dataset']['dataset_type'] = DatasetType.TEST_TORCH_DATASET
        sanity_checked_trainer_config['dataset']['dataset_args']['nsamples'] = self.sanity_config['nsamples']
        sanity_checked_trainer_config['dataset']['dataset_args']['shape'] = self.sanity_config['shape']
        sanity_checked_trainer_config['dataset']['dataset_args']['root_folder'] = self.trainer_config.dataset.dataset_args.get('root_folder', None) 

        # Create a version with only a single epoch for initial validation check
        single_epoch_sanity_checked_trainer_config = sanity_checked_trainer_config.copy()
        single_epoch_sanity_checked_trainer_config['train_loop']['epochs'] = 1
        
        assert isinstance(single_epoch_sanity_checked_trainer_config, Dict) and isinstance(sanity_checked_trainer_config, Dict)
        sanity_checked_trainer_config = TrainerConfig(**sanity_checked_trainer_config)
        single_epoch_sanity_checked_trainer_config = TrainerConfig(**single_epoch_sanity_checked_trainer_config)

        return sanity_checked_trainer_config, single_epoch_sanity_checked_trainer_config
    
    def run_sanity_check(self) -> bool:
        """
        Run the sanity check training loop.
        
        Returns:
            bool: True if sanity check passed, False otherwise
        """
        logging.info("First trying to run a epoch with validation...")
        try:
            # Initialize parent Trainer with modified config
            
            super().__init__(self.single_epoch_sanity_checked_trainer_config.model_dump())
            self.turn_on_runtime_validation()
            self.start()
            logging.info("Single epoch with validation completed successfully.")

            logging.info("Starting sanity check training...")
            logging.info("Now turning off runtime validation.")
            logging.info("=" * 60)
            # Only run training and validation, skip test
            # Initialize parent Trainer with modified config
            super().__init__(self.sanity_checked_trainer_config.model_dump())
            self.turn_off_runtime_validation()
            self.start()    
            
            logging.info("=" * 60)
            logging.info("SANITY CHECK COMPLETED SUCCESSFULLY")
            logging.info("=" * 60)
            
            return True
            
        except Exception as e:
            logging.error("=" * 60)
            logging.error(f"SANITY CHECK FAILED: {str(e)}")
            logging.error("=" * 60)
            raise e
        
    def _start_test_eval(self):
        """
        Override _start_test_eval() to skip test evaluation in sanity check mode.
        """
        pass  # Skip test evaluation


def run_sanity_check(config: TrainerConfig) -> bool:
    """
    Convenience function to run a sanity check.
    
    Args:
        config: Original trainer configuration
        sanity_config: Optional dict to override sanity check parameters
        
    Returns:
        bool: True if sanity check passed, False otherwise
        
    Example:
        >>> from packages.train.sanity_checker import run_sanity_check
        >>> config = {...}  # Your trainer config
        >>> 
        >>> # Use default sanity config
        >>> if run_sanity_check(config):
        >>>     print("Sanity check passed!")
        >>> 
        >>> # Or customize sanity config
        >>> custom_sanity = {
        >>>     'train_split': 0.1,
        >>>     'val_split': 0.1,
        >>>     'epochs': 3
        >>> }
        >>> if run_sanity_check(config, custom_sanity):
        >>>     print("Custom sanity check passed!")
    """
    checker = SanityChecker(config)
    return checker.run_sanity_check()