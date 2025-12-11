from packages.data_objects.dataset import TorchH5Dataset, autoencoder_unpack_func
from packages.io.torch_dataloaders import get_data_loaders
dataset = TorchH5Dataset(h5_path='/media/lolly/SSD/motor_eeg_dataset/motor_eeg_dataset.h5', unpack_func=autoencoder_unpack_func)
train_loader, _, _ = get_data_loaders(dataset, norm_axes=(0,2), target_norm_axes=(0,2), norm_convergence_threshold=1e-3, batch_size=64)

batch = next(iter(train_loader))

from packages.models.vqae_light import VQAE23, VQAE23Config
config = VQAE23Config(
    use_quantizer=False,
    use_cwt=True,
    chunk_samples=160
)
model = VQAE23(config)
import torch
with torch.no_grad():
    out = model(batch['input'])
    cwt = model.cwt_head(batch['input'])
    target = batch['target']
    print("cwt mean:", cwt.mean().item(), "std:", cwt.std().item())
    print("recon mean:", out['reconstruction'].mean().item(), "std:", out['reconstruction'].std().item())
    print("target mean:", target.mean().item(), "std:", target.std().item())