import numpy as np

from packages.data_objects.dataset import Dataset
from packages.io.input_loader import get_cv_loaders_with_static_test, get_data_loaders

data_path = "packages/test/test_output"
dataset = Dataset(root_folder=data_path, file_type="npz", unpack_func="dict")
train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size=8)

for batch in train_loader:
    print(
        f"Batch shape: {batch.shape}"
    )  # Assuming each batch is a dictionary with a 'data' key
    break

for fold, (train_loader, val_loader, test_loader) in enumerate(
    get_cv_loaders_with_static_test(dataset, batch_size=8, n_splits=5, test_size=0.2)
):
    print(f"Fold {fold}:")
    # Use train_loader, val_loader, test_loader as needed
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
