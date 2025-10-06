import torch
import logging
from tqdm import tqdm
from packages.train.helpers import EarlyStopping, BackupManager, History, NoOpHistory
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]  # ensures output to stdout
)

TRAIN_CONFIG = {
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 50,
    'backup_interval': 10,
    'EarlyStopping' : {'patience': 5, 'min_delta': 0.0},
    'BackupManager': {'backup_interval': 10, 'backup_path': './model_backups'},
    'ReduceLROnPlateau': {'mode': 'min', 'patience': 5, 'factor': 0.1, 'verbose': True},
    'history_plot': {'extended': False, 'save_path': './training_history'}
}

METRICS = {
    'MAE': None
}

def _check_precision(model, *data_loaders):
    """
    Ckecks if model and dataloaders have the same precision
    """
    model.eval()
    with torch.no_grad():
        for loader in data_loaders:
            for batch in loader:
                batch = batch.to(next(model.parameters()).device)
                output = model(batch)
                if output.dtype != batch.dtype:
                    logging.warning(f"Precision mismatch: model output dtype {output.dtype}, input dtype {batch.dtype}. Converting model to {batch.dtype}.")
                    model = model.to(batch.dtype)
                return model
    return model

def _setup_model(model):
    """
    Generic model setup function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def _train_loop(model, train_loader, loss_criterion, optimizer, device, history, task_handler):
    """
    Training loop for one epoch
    """
    model.train()
    train_loss = 0.0
    
    for batch in tqdm(train_loader):
            
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs, loss = task_handler._process(loss_criterion, model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * outputs.size(0)

    train_loss /= len(train_loader.dataset)
    task_handler._end_epoch(len(train_loader.dataset))
    history.log_train(train_loss, task_handler.evals)

def _eval_loop(model, val_loader, loss_criterion, device, history, task_handler):
    """
    Evaluation loop for one epoch
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs, loss = task_handler._process(loss_criterion, model, batch)
            val_loss += loss.item() * outputs.size(0)
    val_loss /= len(val_loader.dataset)
    task_handler._end_epoch(len(val_loader.dataset))
    history.log_val(val_loss, task_handler.evals)
    return val_loss

def test_model(model, test_loader, loss_func, metrics: Dict[str, Callable], device):
    model.eval()
    test_loss = 0.0
    metric_eval = {metric : 0 for metric in metrics}
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_func(outputs, batch)
            test_loss += loss.item() * batch.size(0)
            for metric_name, metric_func in metrics.items():
                metric_eval[metric_name] += metric_func(outputs, batch)

    test_loss /= len(test_loader.dataset)
    logging.info(f"Test Loss: {test_loss:.4f}")
    for metric_name, metric_value in metric_eval.items():
        logging.info(f"Test {metric_name}: {metric_value:.4f}")
    return test_loss

def train_model(model, train_loader, val_loader, loss_criterion, optimizer, metrics: Dict[str, Callable] = METRICS, config=TRAIN_CONFIG):
    """
    Main train model function, 
    """
    epochs = config.get('epochs')
    
    if 'history_plot' in config:
        history_save_path = config.get('history_plot', {}).get('save_path')
        plot_type = config.get('history_plot', {}).get('plot_type')
        history = History(save_path=history_save_path, plot_type=plot_type, metrics=metrics)
    else:
        history = NoOpHistory(metrics=metrics)

    model, device = _setup_model(model)
    model = _check_precision(model, train_loader, val_loader)

    optimizer = optimizer(model.parameters(), lr=config.get('lr'))

    if isinstance(loss_criterion, type): 
        loss_criterion = loss_criterion()
    
    for metric_name, metric in metrics.items():
        if isinstance(metric, type):
            metrics[metric_name] = metric()

    helper_handler = HelperHandler(config, optimizer)

    task_handler = TaskHandler(loader=train_loader, metrics=metrics)

    for epoch in range(epochs):
        task_handler._reset_metrics()
        _train_loop(model, train_loader, loss_criterion, optimizer, device, history, task_handler)
        val_loss = _eval_loop(model, val_loader, loss_criterion, device, history, task_handler)

        helper_handler._update_helpers(model, epoch, val_loss)

        if helper_handler.early_stopper.early_stop:
            break

    history.plot_history()
    
class HelperHandler:
    def __init__(self, config, optimizer):
        self.backup_manager = None
        self.early_stopper = None
        self.lr_scheduler = None
        self._last_lr = config.get('lr')
        self._initialize_helpers(config, optimizer)

    def _update_helpers(self, model, epoch, val_loss):
            if self.backup_manager is not None:
                self.backup_manager(model, epoch, val_loss)
            if self.early_stopper is not None:
                self.early_stopper(val_loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)
                if self._last_lr != self.lr_scheduler.get_last_lr()[0]:
                    self._last_lr = self.lr_scheduler.get_last_lr()[0]
                    logging.info(f"Updated Learning Rate: {self.lr_scheduler.get_last_lr()[0]:.6f}")
    
    def _initialize_helpers(self, config, optimizer):
        for helper in config:
            if helper == 'BackupManager':
                self.backup_manager = BackupManager(**config.get('BackupManager', {}))
            elif helper == 'EarlyStopping':
                self.early_stopper = EarlyStopping(**config.get('EarlyStopping', {}))
            elif helper == 'ReduceLROnPlateau':
                self.lr_scheduler = ReduceLROnPlateau(optimizer, **config.get('ReduceLROnPlateau', {}))
                

class TaskHandler:
    def __init__(self, loader, metrics):
        self.handler = None
        self.loader = loader
        self.metrics = metrics
        self._id_task()
        self._reset_metrics()
    def _id_task(self):
        batch = next(iter(self.loader))
        if isinstance(batch, tuple) and len(batch) == 2:
            # (Input, Labels) case
            self.handler = 2
        else:
            # Input = Output (Autoencoder) case
            self.handler = 1
    
    def _process(self, loss_criterion, model, batch):
        
        if self.handler == 2:
            outputs = model(batch[0])
            loss = loss_criterion(outputs, batch[1])

            for metric_name, metric_func in self.metrics.items():
                eval = metric_func(outputs.detach().cpu(), batch[1])
                self.evals[metric_name] += eval * outputs.size(0)

        elif self.handler == 1:
            outputs = model(batch)
            loss = loss_criterion(outputs, batch)

            for metric_name, metric_func in self.metrics.items():
                eval = metric_func(outputs.detach().cpu(), batch)
                self.evals[metric_name] += eval * outputs.size(0)
        return outputs, loss
    
    def _reset_metrics(self):
        self.evals = {metric_name: 0.0 for metric_name in self.metrics}

    def _end_epoch(self, numsamples):
        self.evals = {metric_name: metric_eval / numsamples for metric_name, metric_eval in self.evals.items()}
