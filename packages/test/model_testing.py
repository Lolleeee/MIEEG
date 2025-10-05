from packages.models.Autoencoder import Conv3DAutoencoder
from packages.plotting.reconstruction_plots import plot_reconstruction_distribution

import torch
import argparse
from packages.data_objects.dataset import Dataset
from packages.io.input_loader import get_test_loader

x = torch.randn(8, 50, 7, 5, 250)


# --- Argument parser setup ---
parser = argparse.ArgumentParser(description="Test Conv3DAutoencoder")
parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of embedding")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for input tensor")
args = parser.parse_args()

embedding_dim = args.embedding_dim
epochs = args.epochs
batch_size = args.batch_size

model = Conv3DAutoencoder(in_channels=50, embedding_dim=embedding_dim)


output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Get embedding
embedding = model.encode(x)
print(f"Embedding shape: {embedding.shape}")

# Verify output shape matches input
assert x.shape == output.shape, "Output shape doesn't match input!"
print("✓ Shape verification passed!")

x = torch.randn(batch_size, 50, 7, 5, 250)

output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Get embedding
embedding = model.encode(x)
print(f"Embedding shape: {embedding.shape}")

# Verify output shape matches input
assert x.shape == output.shape, "Output shape doesn't match input!"
print("✓ Shape verification passed!")

# Calculate and print number of parameters in scientific notation
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:.2e}")

print("Model size (MB):", sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6)

# Dummy training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Single sample for training
train_sample = torch.randn(1, 50, 7, 5, 250)

print("\nStarting dummy training loop...")
model.train()
epochs = 1
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    reconstructed = model(train_sample)
    loss = criterion(reconstructed, train_sample)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("✓ Training loop completed!")

dataset = Dataset.get_test_dataset(root_folder="/home/lolly/Desktop/test/patient1", nsamples=1, file_type='npz', unpack_func='dict')

test_loader = get_test_loader(dataset, batch_size=1, num_workers=4)

print("\nStarting dummy training loop...")
model.train()
epochs = 30
for epoch in range(epochs):

    for batch in test_loader:
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

plot_reconstruction_distribution(batch, reconstructed)