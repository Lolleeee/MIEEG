from pyexpat import model
import sys
import torch
import logging
from tqdm.notebook import tqdm
from packages.train.helpers import EarlyStopping, BackupManager, GradientLogger, HelperHandler, History
from packages.train.loss import TorchLoss
from packages.train.metrics import TorchMetric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Dict, List
from torch.amp.grad_scaler import GradScaler
from packages.train.trainer_config_schema import SanityCheckConfig, TrainerConfig
from packages.io.torch_dataloaders import get_data_loaders
from packages.train.seed import _set_seed
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
# TODO check if random seed works for model initialization

class MetricsHandler:
    def __init__(self, loss: TorchLoss, metrics: List[TorchMetric]):
        
        assert isinstance(loss, TorchLoss), "Loss must be an instance of TorchLoss."
        assert all(isinstance(metric, TorchMetric) for metric in metrics), "All metrics must be instances of TorchMetric."
        
        self.loss: TorchLoss = loss
        self.metrics: Dict[str, TorchMetric] = {metric.name: metric for metric in metrics}
        self.metrics_eval: Dict[str, float] = {}
        self._initialize_metrics_eval_dict()

        self.epoch_samples_count: int = 0
        self._new_epoch_setup(current_epoch=0)

        self.loss_eval: torch.Tensor = None

    def _batch_step(self, outputs, batch, detach: bool = False):

        assert isinstance(batch, dict) and 'input' in batch, "Batch must be a dict containing an 'input' key."
        
        if 'target' in batch:
            target = batch['target']
        else:
            target = batch['input']

        if detach:
            outputs = self.detach_outputs(outputs)

        self.epoch_samples_count += target.size(0)

        batch_loss_eval_dict = self.loss(outputs, batch)
        
        # Main loss value to be backpropagated
        loss_eval = batch_loss_eval_dict['loss']
        
        # Accumulate loss into metrics dict
        for loss_comp_name, value in batch_loss_eval_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
            self.metrics_eval[loss_comp_name] += value * target.size(0)
            
        # Accumulate metrics
        for metric_name, metric in self.metrics.items():
            value = metric(outputs, target).detach().cpu()
            self.metrics_eval[metric_name] += value * target.size(0)
        
        return loss_eval

    def _end_epoch_step(self):
        assert self.epoch_samples_count > 0, "No samples were processed in this epoch."
        assert all(value >= 0 for value in self.metrics_eval.values()), "One or more metric evaluations are negative, which is invalid."
        # Scaling metrics by number of samples
        
        self.metrics_eval = self.scale_loss_dict(self.metrics_eval, 1.0 / self.epoch_samples_count, ignore_keys=['epoch'])

        
    def _new_epoch_setup(self, current_epoch: int):
        assert current_epoch is not None, "Current epoch must be provided for new epoch setup."

        self.epoch_samples_count = 0
        for metrics_name in self.metrics_eval:
            if metrics_name == "epoch":
                self.metrics_eval[metrics_name] = current_epoch
            else:
                self.metrics_eval[metrics_name] = 0.0

    def _initialize_metrics_eval_dict(self):
        self.metrics_eval['epoch'] = 0

        for metric in self.metrics.values():
            self.metrics_eval[metric.name] = 0.0

        for key in self.loss.expected_loss_keys:
            self.metrics_eval[key] = 0.0

    @staticmethod
    def detach_outputs(outputs):
        if isinstance(outputs, tuple):
            # Return a new tuple with detached tensors where needed
            return tuple(
                output.detach().cpu() if isinstance(output, torch.Tensor) and output.requires_grad else output
                for output in outputs
            )
        else:
            if isinstance(outputs, torch.Tensor) and outputs.requires_grad:
                return outputs.detach().cpu()
            return outputs
    
    @staticmethod
    def scale_loss_dict(loss_dict: dict, factor: float, ignore_keys=None) -> dict:
        """Multiply each scalar in a loss dict by a constant factor."""
        if ignore_keys is None:
            ignore_keys = []

        out_dict = {k: v * factor if k not in ignore_keys else v for k, v in loss_dict.items()}
        return out_dict



class Trainer():
    def __init__(self, config: Dict):
        self.config = TrainerConfig(**config)

        self._config_unpacking()

        _set_seed(self.config)

        self._classes_init()
        
        self._data_loaders_init()

        self._pretrain_setup()

        self._components_runtime_validation_setup()

        self._call_sanity_checker()

    def _config_unpacking(self):
        self.model = self.config.get_model_class
        assert isinstance(self.model, type) and issubclass(self.model, torch.nn.Module)

        self.loss_criterion = self.config.get_loss_class
        assert isinstance(self.loss_criterion, type) and issubclass(self.loss_criterion, TorchLoss)
        
        self.optimizer = self.config.optimizer.get_optimizer_class
        assert isinstance(self.optimizer, type) and issubclass(self.optimizer, torch.optim.Optimizer)

        self.metrics = self.config.info.get_metric_classes
        assert isinstance(self.metrics, list)
        assert all(isinstance(m, type) and issubclass(m, TorchMetric) for m in self.metrics)

        self.dataset = self.config.dataset.get_dataset_class
        assert isinstance(self.dataset, type) and issubclass(self.dataset, torch.utils.data.Dataset)

        self.device = self.config.get_device
        assert isinstance(self.device, torch.device)

    def _classes_init(self):

        # Init model
        assert isinstance(self.model, type) and issubclass(self.model, torch.nn.Module)
        self.model = self.model(**self.config.model.model_kwargs)
        assert isinstance(self.model, torch.nn.Module)
    
        # Init loss criterion
        self.loss_criterion = self.loss_criterion(**self.config.loss.loss_kwargs)
        assert isinstance(self.loss_criterion, TorchLoss)

        # Init metrics
        self.metrics = [m() for m in self.metrics]
        assert all(isinstance(m, TorchMetric) for m in self.metrics)

        # Init dataset
        self.dataset = self.dataset(**self.config.dataset.dataset_args)
        assert isinstance(self.dataset, torch.utils.data.Dataset)

        
    def _optimizer_init(self):
        assert isinstance(self.optimizer, type) and issubclass(self.optimizer, torch.optim.Optimizer)
        if self.config.optimizer.asym_lr is not None:
            asym_lr = self.config.optimizer.asym_lr
            self.optimizer = self.optimizer(
                self.model.parameters(),
                lr=asym_lr,
                weight_decay=self.weight_decay
            )
            mean_lr = sum(asym_lr.values()) / len(asym_lr)
            self.lr = mean_lr
        else:
            self.lr = self.config.optimizer.lr 
            self.optimizer = self.optimizer(
            self.model.parameters(),
            lr = self.lr,
            weight_decay=self.config.optimizer.weight_decay
        )
        
    def _data_loaders_init(self):
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            dataset=self.dataset,
            sets_size=self.config.dataset.data_loader.set_sizes.model_dump(),
            batch_size=self.config.dataset.data_loader.batch_size,
            norm_axes=self.config.dataset.data_loader.norm_axes,
            target_norm_axes=self.config.dataset.data_loader.target_norm_axes
        )

    def _components_runtime_validation_setup(self):
        """
        Setup components for runtime validation if enabled in config.
        """

        self._validation_components_list: List[Callable] = []

        self._validation_components_list.append(self.loss_criterion)

        for metric in self.metrics:
            self._validation_components_list.append(metric)
        
        assert all(hasattr(component, 'enable_validation') and callable(getattr(component, 'enable_validation')) for component in self._validation_components_list), "All components must have an 'enable_validation' method."
        

    def _call_sanity_checker(self):
        if self.config.sanity_check is not None and self.config.sanity_check.enabled:
            logging.info("Initializing Sanity Checker...")
            from packages.train.sanity_check import run_sanity_check
            
            result = run_sanity_check(self.config)
            if not result:
                raise RuntimeError("Sanity check failed. Aborting training.")
            
    def _pretrain_setup(self):
        """
        Function which executes all the thing that should happen before starting any loop
        Things like instance creation and parameters setups
        """
        # Setting model to specified device
        self.model.to(self.device)

        # Initialize optimizer with lr and weight decay
        self._optimizer_init()

        self.metrics_handler = MetricsHandler(self.loss_criterion, self.metrics)

        self.helper_handler = HelperHandler(self)

        


    def start(self):

        self._start_train_loop()

        self._start_test_eval()

        self.helper_handler._call_end_train_step()

    def _start_train_loop(self):
        #THINGS
        #TRACKER AND SCHEDULERS UPDATES ABOUT TRAINING
        self.model.train()

        self.use_amp = self.config.gradient_control.use_amp
        self.grad_clip = self.config.gradient_control.grad_clip
        self.epochs = self.config.train_loop.epochs
        with tqdm(desc="Epochs", total=self.epochs, position=1, leave=True) as Epochpbar:
            for epoch in range(self.epochs):

                self.metrics_handler._new_epoch_setup(epoch)

                self.model.train()
                print("model.train")
                train_metrics = self._train_epoch()
                
                self.metrics_handler._new_epoch_setup(epoch)
                
                val_metrics = self._start_val_loop()
                
                if 'early_stopping' in self.helper_handler.helpers and self.helper_handler.helpers['early_stopping'].stop_early:
                    logging.info("Early stopping triggered. Ending training loop.")
                    break

                Epochpbar.update()
                Epochpbar.set_postfix({'Epoch Train Loss': train_metrics.get('loss', "No loss found"), 'Epoch Val Loss': val_metrics.get('loss', "No loss found")})

    def _train_epoch(self):
        
        self.scaler = GradScaler(enabled=self.use_amp)

        with tqdm(desc="Training Batches", total=len(self.train_loader), position=1, leave=True) as Batchpbar:
            for batch in self.train_loader:
                
                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(batch['input'])
                    batch_loss_eval = self.metrics_handler._batch_step(outputs, batch)
                    
                self.scaler.scale(batch_loss_eval).backward()

                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                Batchpbar.update()
                Batchpbar.set_postfix({'Batch Loss': batch_loss_eval.item()})

            self.metrics_handler._end_epoch_step()

            self.helper_handler._call_end_train_epoch_step(self.metrics_handler.metrics_eval)

        return self.metrics_handler.metrics_eval
    
    def _start_val_loop(self):
        print("model.eval")
        self.model.eval()

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(batch['input'])
                
                self.metrics_handler._batch_step(outputs, batch, detach=True)
                
        self.metrics_handler._end_epoch_step()
        
        self.helper_handler._call_end_val_step(self.metrics_handler.metrics_eval)

        return self.metrics_handler.metrics_eval
    
    def _start_test_eval(self):

        self.metrics_handler._new_epoch_setup(current_epoch=0)

        self.model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['input'])

                self.metrics_handler._batch_step(outputs, batch, detach=True)

        self.metrics_handler._end_epoch_step()

        test_metrics = self.metrics_handler.metrics_eval
        for metric_name, metric_value in test_metrics.items():
            if metric_name != "epoch":
                logging.info(f"Test {metric_name}: {metric_value:.4f}")
        
        self.metrics_handler._end_epoch_step()
    
    def turn_on_runtime_validation(self):
        """
        Turn on validation in all relevant components.
        """

        for component in self._validation_components_list:
            component.enable_validation()
        

    def turn_off_runtime_validation(self):
        """
        Turn off validation in all relevant components.
        """
        for component in self._validation_components_list:
            component.disable_validation()



