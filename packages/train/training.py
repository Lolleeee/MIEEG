import sys
import torch
import logging
from tqdm.notebook import tqdm
from packages.train.helpers import EarlyStopping, BackupManager, History, NoOpHistory
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Dict, List


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

TRAIN_CONFIG = {
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 50,
    'backup_interval': 10,
    'EarlyStopping' : {'patience': 5, 'min_delta': 0.0},
    'BackupManager': {'backup_interval': 10, 'backup_path': './model_backups'},
    'ReduceLROnPlateau': {'mode': 'min', 'patience': 5, 'factor': 0.1, 'verbose': True},
    'history_plot': {'extended': False, 'save_path': './training_history'}
}

def _check_precision(model, *data_loaders):
    """
    Checks if model and dataloaders have the same precision
    """
    model_dtype = next(model.parameters()).dtype
    for dataloader in data_loaders:
        batch = next(iter(dataloader))
        if batch.dtype != model_dtype:
            logging.warning(f"DataLoader dtype {batch.dtype} does not match model dtype {model_dtype}. Converting model to {batch.dtype}.")
            model = model.to(batch.dtype)
            break  # No need to check further batches
    return model

# TODO Have it a separate module maybe?
class TaskHandler:
    def __init__(self, loader = None, metrics = None, batch_size = None):

        self.batch_size = batch_size
        self.loader = loader

        self.metrics = metrics if metrics is not None else {}

        self._reset_metrics()
    
    def process(self, loss_criterion, model, batch):
        
        outputs = model(batch)
        loss = loss_criterion(outputs, batch)

        self._eval_metrics(outputs, batch)

        return outputs, loss
    
    def _eval_metrics(self, outputs, batch):  

        outputs = detach_outputs(outputs)

        for metric_name, metric_func in self.metrics.items():
            eval = metric_func(outputs, batch)
            self.evals[metric_name] += eval * self.batch_size

    def _reset_metrics(self):
        self.evals = {metric_name: 0.0 for metric_name in self.metrics}

    def _end_epoch(self, numsamples):
        self.evals = {metric_name: metric_eval / numsamples for metric_name, metric_eval in self.evals.items()}


def _setup_model(model):
    """
    Generic model setup function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def _train_loop(model, train_loader, loss_criterion, optimizer, device, history, task_handler: TaskHandler):
    print(tqdm.__module__)
    model.train()
    train_loss = 0.0

    with tqdm(desc="Training Batches", total=len(train_loader), position=1, leave=True) as batchpbar:
        for batch in train_loader:
                
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs, loss = task_handler.process(loss_criterion, model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * train_loader.batch_size
            batchpbar.update()

        train_loss /= len(train_loader.dataset)
        task_handler._end_epoch(len(train_loader.dataset))
        history.log_train(train_loss, task_handler.evals)
            

def _eval_loop(model, val_loader, loss_criterion, device, history, task_handler: TaskHandler):
    """
    Evaluation loop for one epoch
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs, loss = task_handler.process(loss_criterion, model, batch)
            val_loss += loss.item() * val_loader.batch_size
    val_loss /= len(val_loader.dataset)
    task_handler._end_epoch(len(val_loader.dataset))
    history.log_val(val_loss, task_handler.evals)
    return val_loss

# TODO Test this function
def test_model(model, test_loader, loss_func, metrics: Dict[str, Callable], device):
    model.eval()
    test_loss = 0.0
    metric_eval = {metric : 0 for metric in metrics}
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = loss_func(outputs, batch)
            test_loss += loss.item() * test_loader.batch_size
            for metric_name, metric_func in metrics.items():
                metric_eval[metric_name] += metric_func(outputs, batch)

    test_loss /= len(test_loader.dataset)
    logging.info(f"Test Loss: {test_loss:.4f}")
    for metric_name, metric_value in metric_eval.items():
        logging.info(f"Test {metric_name}: {metric_value:.4f}")
    return test_loss

# TODO Make this into a class and handle all config options, would be nice to do .train() on it
def train_model(model, train_loader, val_loader, loss_criterion, optimizer, metrics: Dict[str, Callable] = None, config=TRAIN_CONFIG):
    """
    Main train model function, 
    """

    epochs = config.get('epochs')
    batch_size = config.get('batch_size', TRAIN_CONFIG['batch_size'])
    lr = config.get('lr', TRAIN_CONFIG['lr'])
    weight_decay = config.get('weight_decay', TRAIN_CONFIG['weight_decay'])


    if 'history_plot' in config:
        history_save_path = config.get('history_plot', {}).get('save_path')
        plot_type = config.get('history_plot', {}).get('plot_type')
        history = History(save_path=history_save_path, plot_type=plot_type, metrics=metrics)
    else:
        history = NoOpHistory(metrics=metrics)

    model, device = _setup_model(model)

    model = _check_precision(model, train_loader, val_loader)

    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Convert loss criterion and metrics to instances if they are not already
    if isinstance(loss_criterion, type): 
        loss_criterion = loss_criterion()

    for metric_name, metric_func in metrics.items():
        if isinstance(metric_func, type):
            metrics[metric_name] = metric_func()

    helper_handler = HelperHandler(config, optimizer)

    task_handler = TaskHandler(loader=train_loader, metrics=metrics, batch_size=batch_size)

    with tqdm(desc="Epochs", total=epochs, position=1, leave=True) as Epochpbar:
        for epoch in range(epochs):

            task_handler._reset_metrics()

            _train_loop(model, train_loader, loss_criterion, optimizer, device, history, task_handler)

            val_loss = _eval_loop(model, val_loader, loss_criterion, device, history, task_handler)

            helper_handler._update_helpers(model, epoch, val_loss)
            if helper_handler.early_stopper is not None and helper_handler.early_stopper.early_stop:
                break
            Epochpbar.update()

    history.plot_history()
    return model
    

# TODO Move this into Helpers module
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