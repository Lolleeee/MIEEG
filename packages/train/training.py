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
from packages.train.trainer_config_schema import TrainerConfig
from packages.io.torch_dataloaders import get_data_loaders
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)


class MetricsHandler:
    def __init__(self, loss: TorchLoss, metrics: List[TorchMetric]):
        
        assert isinstance(loss, TorchLoss), "Loss must be an instance of TorchLoss."
        assert all(isinstance(metric, TorchMetric) for metric in metrics), "All metrics must be instances of TorchMetric."
        self.metrics = {metric.name: metric for metric in metrics}
        self.loss = loss
        self._new_epoch_setup()

    def _batch_step(self, outputs, batch, detach: bool = False):

        if isinstance(batch, tuple) or isinstance(batch, list):
            batch_size = batch[0].size(0)
        else:
            batch_size = batch.size(0)

        if detach:
            outputs = self.detach_outputs(outputs)

        self.samples_count += batch_size

        self.batch_loss_eval = self.loss(outputs, batch)
        assert self.batch_loss_eval.item() >= 0 and not torch.isnan(self.batch_loss_eval) and not torch.isinf(self.batch_loss_eval), \
            f"Batch loss is invalid with value {self.batch_loss_eval.item()}."

        self.epoch_loss += self.batch_loss_eval.item() * batch_size

        for metric_name, metric in self.metrics.items():
            eval = metric(outputs, batch)

            self.metrics_eval[metric_name] += eval * batch_size

        return self.batch_loss_eval

    def _end_epoch_step(self):
        assert self.samples_count > 0, "No samples were processed in this epoch."
        assert self.epoch_loss >= 0, "Epoch loss is negative, which is invalid."
        assert all(value >= 0 for value in self.metrics_eval.values()), "One or more metric evaluations are negative, which is invalid."

        self.epoch_loss /= self.samples_count
        self.epoch_metrics = {metric_name: metric_eval / self.samples_count for metric_name, metric_eval in self.metrics_eval.items()}

    def _new_epoch_setup(self):
        self.epoch_loss = 0.0
        self.metrics_eval = {metric_name: 0.0 for metric_name in self.metrics}
        self.samples_count = 0


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
    



class Trainer():
    def __init__(self, config: Dict):
        self.config = TrainerConfig(**config)

        self._config_unpacking()

        self._classes_init()

        self._data_loaders_init()

        self._pretrain_setup()

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
        )

    def _pretrain_setup(self):
        """
        Function which executes all the thing that should happen before starting any loop
        Things like instance creation and parameters setups
        """
        # Setting model to specified device
        self.model.to(self.device)

        # Initialize optimizer with lr and weight decay
        self._optimizer_init()

        self.helper_handler = HelperHandler(self.config, self.model, self.optimizer)

        self.metrics_handler = MetricsHandler(self.loss_criterion, self.metrics)

        self.current_epoch = 0

    def start(self):

        self._start_train_loop()

        self._start_test_eval()


    def _start_train_loop(self):
        #THINGS
        #TRACKER AND SCHEDULERS UPDATES ABOUT TRAINING
        self.model.train()

        self.use_amp = self.config.gradient_control.use_amp
        self.grad_clip = self.config.gradient_control.grad_clip

        with tqdm(desc="Epochs", total=self.config.train_loop.epochs, position=1, leave=True) as Epochpbar:
            for epoch in range(self.config.train_loop.epochs):

                epoch_loss, epoch_metrics = self._train_epoch()

                self.helper_handler._call_end_train_epoch_step(epoch, epoch_loss, epoch_metrics)

                val_loss, val_metrics = self._start_val_loop()

                self.helper_handler._call_end_val_step(epoch, val_loss, val_metrics)
                
                if self.helper_handler.stop_early:
                    logging.info("Early stopping triggered. Ending training loop.")
                    break

                Epochpbar.update()
                Epochpbar.set_postfix({'Epoch Loss': val_loss})

    def _train_epoch(self):
        self.metrics_handler._new_epoch_setup()
        
        self.scaler = GradScaler(enabled=self.use_amp)

        with tqdm(desc="Training Batches", total=len(self.train_loader), position=1, leave=True) as Batchpbar:
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(batch)
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
            epoch_loss = self.metrics_handler.epoch_loss
            self.current_epoch += 1
            return epoch_loss, self.metrics_handler.epoch_metrics
        
    def _start_val_loop(self):
        #TRACKED AND SCHEDULERS UPDATE ABOUT VALIDATION
        self.metrics_handler._new_epoch_setup()

        self.model.eval()

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)

                outputs = self.model(batch)

                val_loss = self.metrics_handler._batch_step(outputs, batch, detach=True)

        return val_loss, self.metrics_handler.epoch_metrics
    
    def _start_test_eval(self):

        self.metrics_handler._new_epoch_setup()

        self.model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)

                self.metrics_handler._batch_step(outputs, batch, detach=True)
        test_loss = self.metrics_handler.epoch_loss
        test_metrics = self.metrics_handler.epoch_metrics
        logging.info(f"Test Loss: {test_loss:.4f}")
        for metric_name, metric_value in test_metrics.items():
            logging.info(f"Test {metric_name}: {metric_value:.4f}")
        


