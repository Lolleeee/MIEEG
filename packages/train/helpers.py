import logging
import os
from typing import Callable, Dict

import matplotlib.pyplot as plt
import torch

from packages.plotting import train_plots

# TODO integrate better with training loop updates. Make a stage like loop, like in stage initial training, during, after, late and such
class BackupManager:
    def __init__(self, backup_interval, backup_path):
        self.backup_interval = backup_interval
        self.backup_path = backup_path
        self.last_backup_epoch = -1
        os.makedirs(backup_path, exist_ok=True)
        self.best_metric = None

        self.best_model_backup_file = ""

    def __call__(self, model, epoch, metric):
        self._best_model_backup(model, metric, epoch)
        self._periodic_backup(epoch, model)

    def _best_model_backup(self, model, metric, epoch):
        if self.best_metric is None or metric < self.best_metric:
            backup_file = os.path.join(
                self.backup_path, f"best_model_epoch_{epoch + 1}.pt"
            )
            torch.save(model.state_dict(), backup_file)
            logging.info(
                f"Best model saved at epoch {epoch + 1} with metric {metric:.4f}"
            )
            self.best_model_backup_file = backup_file
            self.best_metric = metric
            self.last_backup_epoch = epoch

    def _periodic_backup(self, epoch, model):
        if (epoch + 1) % self.backup_interval == 0 and epoch != self.last_backup_epoch:
            backup_file = os.path.join(self.backup_path, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), backup_file)
            logging.info(f"Periodic backup saved at epoch {epoch + 1}")
            self.last_backup_epoch = epoch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info("Early stopping triggered.")


class History:
    def __init__(
        self,
        save_path: str = None,
        plot_type: str = None,
        metrics: Dict[str, Callable] = None,
    ):
        self.train_history = {'loss': []}
        self.val_history = {'loss': []}
        self.save_path = save_path
        self.plot_type = plot_type
        self.metrics = metrics if metrics is not None else {}
        self._init_histories()

    def _init_histories(self):
        for metric_name in self.metrics:
            self.train_history[metric_name] = []
            self.val_history[metric_name] = []

    def log_train(self, loss: float, metrics_eval: Dict[str, float]):
        self.train_history['loss'].append(loss)
        log_string = f"Train Loss: {loss:.4f}"
        for metric_name in self.metrics:
            self.train_history[metric_name].append(metrics_eval.get(metric_name))
            log_string += f", {metric_name}: {metrics_eval.get(metric_name):.4f}"

        logging.info(log_string)

    def log_val(self, loss: float, metrics_eval: Dict[str, float]):
        self.val_history['loss'].append(loss)
        log_string = f"Val Loss: {loss:.4f}"
        for metric_name in self.metrics:
            self.val_history[metric_name].append(metrics_eval.get(metric_name))
            log_string += f", {metric_name}: {metrics_eval.get(metric_name):.4f}"

        logging.info(log_string)

    def plot_history(self):
        if self.plot_type == "extended":
            train_plots.plot_history_extended(self)
            self.save_plot()
        elif self.plot_type == "tight":
            train_plots.plot_history_tight(self)
            self.save_plot()
        elif self.plot_type is None:
            self.save_plot()
            plt.close()
        else:
            raise ValueError(
                "Invalid value for 'plot_type'. Choose from 'extended', 'tight', or None."
            )

    def save_plot(self):
        if self.save_path:
            plt.savefig(self.save_path)
            logging.info(f"Training history plot saved to {self.save_path}")

class NoOpHistory(History):
    def __init__(self):
        super().__init__
    def plot_history(self, *args, **kwargs):
        pass

class GradientLogger:
    def __init__(self, interval= None):
        self.interval = interval
        self.counter = 0

    def log(self, model):
        if self.interval is None:
            return
        self.counter += 1
        if self.counter % self.interval == 0:
            for name, param in model.named_parameters():
                logging.info(f"Gradient norm for {name}: {param.grad.norm().item():.4f}")