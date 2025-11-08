import logging
import os
from re import match
from typing import Callable, Dict

import matplotlib.pyplot as plt
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau
from packages.plotting import train_plots
from packages.train.metrics import TorchMetric
from packages.train.trainer_config_schema import HistoryPlotSchema, PlotType
from packages.train.loss import TorchLoss
from typing import List
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.train.training import Trainer

class Helper():
    """
    Base class for all helpers.
    Helpers are implemented by implementing at least one of the following methods:
    - _end_train_epoch_step(self, **kwargs)
    - _end_val_step(self, **kwargs)
    These methods will be called at the end of each training epoch and validation step respectively.
    kwargs can contain any relevant information such as epoch number, loss values, metrics, etc.
    """
    def __init__(self):
        self.name = self.__class__.__name__
        pass

class BackupManager(Helper):
    def __init__(self, backup_interval, backup_path, model, metric = 'loss', mode: str = 'min'):
        self.backup_interval = backup_interval
        self.backup_path = backup_path
        self.last_backup_epoch = -1
        self.metric_name = metric

        os.makedirs(backup_path, exist_ok=True)
        self.best_model_backup_file = ""

        self.mode = mode

        if mode == 'min':
            self.best_metric = float('inf')
        elif mode == 'max':
            self.best_metric = float('-inf')
        else:
            raise ValueError("mode must be either 'min' or 'max'")
        
        self.model = model

    def _end_val_step(self, **kwargs):
        epoch = kwargs.get("epoch")
        metric = kwargs[self.metric_name]
        self._best_model_backup(metric, epoch)
        self._periodic_backup(epoch)

    def _best_model_backup(self, metric, epoch):
        if (self.mode == 'min' and metric < self.best_metric) or (self.mode == 'max' and metric > self.best_metric):
            backup_file = os.path.join(
                self.backup_path, f"best_model_epoch_{epoch + 1}_{time.time()}.pt"
            )
            torch.save(self.model.state_dict(), backup_file)
            logging.info(
                f"Best model saved at epoch {epoch + 1} with {self.metric_name}: {metric:.4f}"
            )
            self.best_model_backup_file = backup_file
            self.best_metric = metric
            self.last_backup_epoch = epoch

    def _periodic_backup(self, epoch):
        if (epoch + 1) % self.backup_interval == 0 and epoch != self.last_backup_epoch:
            backup_file = os.path.join(self.backup_path, f"model_epoch_{epoch + 1}_{time.time()}.pt")
            torch.save(self.model.state_dict(), backup_file)
            logging.info(f"Periodic backup saved at epoch {epoch + 1}")
            self.last_backup_epoch = epoch

class EarlyStopping(Helper):
    def __init__(self, patience=5, min_delta=0.0, metric='loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_metric = float("inf") if mode == 'min' else float("-inf")
        self.counter = 0
        self.stop_early = False

    def _end_val_step(self, **kwargs):
        metric = kwargs[self.metric.name]
        if (self.mode == 'min' and metric < self.best_metric - self.min_delta) or (self.mode == 'max' and metric > self.best_metric + self.min_delta):
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_early = True
                


class History(Helper):
    def __init__(
        self,
        save_path: str,
        plot_type: PlotType,
        metrics: Dict[str, float],
    ):
        self.save_path = save_path
        self.plot_type = plot_type
        
        self._init_histories(metrics)

    def _init_histories(self, metrics: Dict[str, float]):
        self.metrics = list(metrics.keys())
        self.train_history = {name: [] for name in self.metrics}
        self.val_history = {name: [] for name in self.metrics}

    def _end_train_epoch_step(self, **kwargs):
        self.train_history, log_string = self._log_history(
            curr_history=self.train_history,
            metrics = kwargs)

        logging.info("Training" + log_string)

    def _end_val_step(self, **kwargs):
        self.val_history, log_string = self._log_history(
            curr_history=self.val_history,
            metrics = kwargs)
        
        logging.info("Validation" + log_string)

    def _end_train_step(self):

        self._plot_history()

    @staticmethod
    def _log_history(curr_history: Dict[str, List[float]], metrics: Dict[str, float]):
        log_string = ""
        for metric_name, metric_value in metrics.items():
            curr_history[metric_name].append(metric_value)  # Debug print
            log_string += f"--- {metric_name}: {metric_value:.4f}"

        return curr_history, log_string

    def _plot_history(self):
        match(self.plot_type):
            case PlotType.EXTENDED:
                train_plots.plot_history_extended(self)
                self.save_plot()
            case PlotType.TIGHT:
                train_plots.plot_history_tight(self)
                self.save_plot()
            case PlotType.OFF:
                return
            case _:
                raise ValueError(
                    "Invalid value for 'plot_type'. Choose from 'extended', 'tight', or None."
                )

    def save_plot(self):
        if self.save_path:
            plt.savefig(self.save_path)
            logging.info(f"Training history plot saved to {self.save_path}")

# TODO Do both batch and epoch tracking and setup a better visualization
class GradientLogger(Helper):
    def __init__(self, model, interval= None):
        self.interval = interval
        self.counter = 0
        self.model = model

    def _end_train_epoch_step(self, **kwargs):
        if self.interval is None:
            return
        self.counter += 1
        if self.counter % self.interval == 0:
            for name, param in self.model.named_parameters():
                logging.info(f"Gradient norm for {name}: {param.grad.norm().item():.4f}")


class LRScheduler(Helper):
    def __init__(self, scheduler: ReduceLROnPlateau):
        self.scheduler = scheduler
        self._last_lr = None

    def _end_val_step(self, **kwargs):
        loss = kwargs["loss"]

        self.scheduler.step(loss)
        if self._last_lr != self.scheduler.get_last_lr()[0]:
            self._last_lr = self.scheduler.get_last_lr()[0]
            logging.info(f"Updated Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")




class HelperHandler:
    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer
        self._initialize_helpers()

    def _call_end_train_epoch_step(self, metrics):
        for helper_instance in self.helpers.values():
            if hasattr(helper_instance, '_end_train_epoch_step'):
                helper_instance._end_train_epoch_step(**metrics)

    def _call_end_val_step(self, metrics):
        for helper_instance in self.helpers.values():
            if hasattr(helper_instance, '_end_val_step'):
                helper_instance._end_val_step(**metrics)

    def _call_end_train_step(self):
        for helper_instance in self.helpers.values():
            if hasattr(helper_instance, '_end_train_step'):
                helper_instance._end_train_step()

    def _initialize_helpers(self):
        all_helpers = self.trainer.config.helpers.model_dump().keys()

        selected_helpers = [
            helper_name for helper_name in all_helpers
            if getattr(self.trainer.config.helpers, helper_name) is not None
        ]

        self.helpers = {}
        for helper_name in selected_helpers:
            match helper_name:
                case "history_plot":
                    helper_instance = History(
                        save_path=self.trainer.config.helpers.history_plot.save_path,
                        plot_type=self.trainer.config.helpers.history_plot.plot_type,
                        metrics=self.trainer.metrics_handler.metrics_eval
                    )
                case "reduce_lr_on_plateau":
                    helper_instance = LRScheduler(
                        scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(
                            self.trainer.optimizer,
                            **self.trainer.config.helpers.reduce_lr_on_plateau.model_dump()
                        )
                    )
                case "gradient_logger":
                    helper_instance = GradientLogger(
                        model=self.trainer.model,
                        interval=self.trainer.config.helpers.gradient_logger.interval
                    )
                case "backup_manager":
                    helper_instance = BackupManager(
                        backup_interval=self.trainer.config.helpers.backup_manager.backup_interval,
                        backup_path=self.trainer.config.helpers.backup_manager.backup_path,
                        model=self.trainer.model,
                        metric=self.trainer.config.helpers.backup_manager.metric,
                        mode=self.trainer.config.helpers.backup_manager.mode
                    )
                case "early_stopping":
                    helper_instance = EarlyStopping(
                        patience=self.trainer.config.helpers.early_stopping.patience,
                        min_delta=self.trainer.config.helpers.early_stopping.min_delta,
                        metric=self.trainer.config.helpers.early_stopping.metric,
                        mode=self.trainer.config.helpers.early_stopping.mode
                    )

                case _:
                    logging.warning(f"Unknown helper: {helper_name}, skipping initialization.")
                
                    continue
            
            logging.info(f"Initialized helper: {helper_name}")

            self.helpers[helper_name] = helper_instance
