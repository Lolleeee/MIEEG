import torch
from packages.data_objects.dataset import TorchDataset
from packages.io.torch_dataloaders import get_data_loaders
from packages.models.vqae_skip import VQAE as VQVAESkip
from packages.models.vqae import SequenceProcessor
import numpy as np
import sys
from sklearn.manifold import TSNE

model = SequenceProcessor(chunk_shape=(25, 7, 5, 32), embedding_dim=64, codebook_size=2048, use_quantizer=False)
model.chunk_ae = VQVAESkip(
    in_channels = 25,
    input_spatial=(7, 5, 32),
    embedding_dim=64,
    use_quantizer=False,
    use_skip_connections=False,
    num_downsample_stages=3)
model_dict = torch.load('model_backups/best_model_epoch_89.pt', map_location='cpu')

model.load_state_dict(model_dict, strict=True)

model.eval()

dataset = TorchDataset("test/test_output/", chunk_size=32)

train_loader, val_loader, _ = get_data_loaders(dataset, sets_size={'train': 0.01, 'val': 0.3}, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = next(iter(val_loader)).to(device)
out = model(x)
from packages.plotting.reconstruction_plots import plot_reconstruction_slices
plot_reconstruction_slices(x[0,0,...], out[0][0,0,...])
loss = torch.nn.MSELoss()(out[0], x)
print(f"Reconstruction loss: {loss.item()}")
print(f"x shape: {x.shape}")
sys.exit(0)
out = model.encode_sequence(x)
for o in out:
    if hasattr(o, 'shape'):
        print(f"Encoded output shape: {o.shape}")
    else:
        print(f"Encoded output: {o}")

embs = out[0].reshape(-1, out[0].shape[-1])

print(f"Embeddings shape: {embs.shape}")

import matplotlib.pyplot as plt

# move to cpu and numpy
emb_np = embs.cpu().detach().numpy()

# optionally subsample if too many points (t-SNE is slow)
max_points = 2000
if emb_np.shape[0] > max_points:
    idx = np.random.choice(emb_np.shape[0], max_points, replace=False)
    emb_np_sample = emb_np[idx]
else:
    emb_np_sample = emb_np

# run t-SNE
tsne = TSNE(n_components=2, perplexity=10, n_iter_without_progress=1000, init='pca', random_state=42)
emb_2d = tsne.fit_transform(emb_np_sample)

# plot
plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=6, alpha=0.7)
plt.title("t-SNE of embeddings")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()