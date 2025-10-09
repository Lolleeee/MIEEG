import argparse

from pyparsing import Dict
import torch

from packages.data_objects.dataset import CustomTestDataset, Dataset
from packages.io.input_loader import get_test_loader


from packages.plotting.reconstruction_plots import plot_reconstruction_distribution
import logging

from packages.train.loss import VaeLoss
from packages.train.training import TaskHandler
logging.basicConfig(level=logging.INFO)


# --- Argument parser setup ---
parser = argparse.ArgumentParser(description="Test Conv3DAutoencoder")
parser.add_argument(
    "--embedding_dim", type=int, default=256, help="Dimension of embedding"
)
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument(
    "--batch_size", type=int, default=8, help="Batch size for input tensor"
)
args = parser.parse_args()

embedding_dim = args.embedding_dim
epochs = args.epochs
batch_size = args.batch_size

class ModelTester:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape
        self.test_forward()
        
        self.is_multiout = self._is_tuple(self.output)

        self.has_encoder = hasattr(self.model, 'encode')
        self.has_decoder = hasattr(self.model, 'decode')

        if self.has_encoder:
            self.sample_encoded = self.model.encode(torch.randn(*self.input_shape))
            self.is_multiembed = self._is_tuple(self.sample_encoded)

    def test_forward(self):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(*self.input_shape)
            self.output = self.model(x)
            return self.output

    def _is_tuple(self, x):
        return isinstance(x , tuple)
    

    def _check_shapes(self):
        if self.is_multiout:
            out_shapes = [o.shape for o in self.output]
            logging.info(f"Output shapes: {out_shapes}")
        else:
            logging.info(f"Output shape: {self.output.shape}")

        if self.has_encoder:
            if self.is_multiembed:
                embed_shapes = [e.shape for e in self.sample_encoded]
                logging.info(f"Encoded shapes: {embed_shapes}")
            else:
                logging.info(f"Encoded shape: {self.sample_encoded.shape}")

        if self.has_decoder and self.has_encoder:
            if self.is_multiembed:
                latent_sample = torch.randn(*self.sample_encoded[0].shape)
            else:
                latent_sample = torch.randn(*self.sample_encoded.shape)
            decoded = self.model.decode(latent_sample)
            logging.info(f"Decoded shape: {decoded.shape}")

    def _get_model_size(self):
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6

    def _get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def model_summary(self):
        self._check_shapes()
        logging.info(f"Number of parameters: {self._get_num_params():.2e}")
        logging.info(f"Model size (MB): {self._get_model_size():.2f}")

    def run_dummy_training_loop(self, criterion, optimizer, dataset_params: Dict=None, device='cpu', epochs=5, batch_size=8):
        if dataset_params is None:
            dataset_params = {}
        dataset = CustomTestDataset(**dataset_params)
        test_loader = get_test_loader(dataset, batch_size=batch_size, num_workers=4)
        task_handler = TaskHandler(loader=test_loader)

        self.model.to(device)

        
        logging.info("\nStarting dummy training loop...")
        self.model.train()
        for epoch in range(epochs):
            for train_sample in test_loader:
                train_sample = train_sample.to(device)
                optimizer.zero_grad()

                outputs, loss = task_handler.process(criterion, self.model, train_sample)
                
                loss.backward()
                optimizer.step()

            if epoch % 2 == 0:
                logging.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        logging.info("✓ Training loop completed!")


def autoencoder_assertions(model, input, output):
    assert model is not None, "Model is None!"
    assert output is not None, "Output is None!"
    assert output.shape == input.shape, f"Output shape {output.shape} doesn't match input shape {input.shape}!"
    logging.info("✓ All assertions passed!")

from packages.models.autoencoder_convnext import Conv3DAE as new_Conv3DAE
# model = Conv3DAE(in_channels=25, embedding_dim=16, hidden_dims=[32, 48])
model1 = new_Conv3DAE(in_channels=25, embedding_dim=16, hidden_dims=[64, 128, 256], use_convnext=False)

print(model1)

from packages.models.autoencoder import Conv3DAE as old_Conv3DAE
model2 = old_Conv3DAE(in_channels=25)

print(model2)
# model_tester = ModelTester(model, (batch_size, 25, 7, 5, 250))
# dataset_params = {'nsamples': 40, 'shape': (25, 7, 5, 250)}
# model_tester.model_summary()
# model_tester.run_dummy_training_loop(VaeLoss(), torch.optim.AdamW(model.parameters(), lr=1e-3), dataset_params=dataset_params, epochs=1)
