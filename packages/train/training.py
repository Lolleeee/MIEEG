import torch
import logging
from tqdm import tqdm
from packages.train.helpers import EarlyStopping, BackupManager, History, NoOpHistory
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Callable, Dict, List

logging.basicConfig(level=logging.INFO)

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

def _train_loop(model, train_loader, criterion, optimizer, device):
    """
    Training loop for one epoch
    """
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.size(0)
    train_loss /= len(train_loader.dataset)
    return train_loss

def _eval_loop(model, val_loader, criterion, device):
    """
    Evaluation loop for one epoch
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            val_loss += loss.item() * batch.size(0)
    val_loss /= len(val_loader.dataset)
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

def train_model(model, train_loader, val_loader, criterion, optimizer, metrics: Dict[str, Callable], config=TRAIN_CONFIG):

    epochs = config.get('epochs')
    
    if 'history_plot' in config:
        history_save_path = config.get('history_plot', {}).get('save_path')
        plot_type = config.get('history_plot', {}).get('plot_type')
        history = History(save_path=history_save_path, plot_type=plot_type)
    else:
        history = NoOpHistory()

    model, device = _setup_model(model)
    model = _check_precision(model, train_loader, val_loader)

    optimizer = optimizer(model.parameters(), lr=config.get('lr'))
    criterion = criterion()
    helper_handler = HelperHandler(config, optimizer)


    for epoch in range(epochs):
        train_loss = _train_loop(model, train_loader, criterion, optimizer, device)
        val_loss = _eval_loop(model, val_loader, criterion, device)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        history.log_train_loss(train_loss)
        history.log_val_loss(val_loss)

        helper_handler._update_helpers(model, epoch, val_loss)

    history.plot_history()
    
class HelperHandler:
    def __init__(self, config, optimizer):
        self.backup_manager = None
        self.early_stopper = None
        self.lr_scheduler = None
        
        self._initialize_helpers(config, optimizer)

    def _update_helpers(self, model, epoch, val_loss):
            if self.backup_manager is not None:
                self.backup_manager(model, epoch, val_loss)
            elif self.early_stopper is not None:
                self.early_stopper(val_loss)
            elif self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)
            else:
                pass.cargo/bin
    
    
    def _initialize_helpers(self, config, optimizer):
        for helper in config:
            if helper == 'BackupManager':
                self.backup_manager = BackupManager(**config.get('BackupManager', {}))
            elif helper == 'EarlyStopping':
                self.early_stopper = EarlyStopping(**config.get('EarlyStopping', {}))
            elif helper == 'ReduceLROnPlateau':
                self.lr_scheduler = ReduceLROnPlateau(optimizer, **config.get('ReduceLROnPlateau', {}))