
"""
Test models for use in training tests.
Simple, lightweight models that can be quickly instantiated.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class SimpleAutoencoder(nn.Module):
    """
    Simple autoencoder for testing.
    Compatible with variable input dimensions and embedding sizes.
    """
    def __init__(
        self,
        in_channels: int = 10,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        **kwargs  # Accept and ignore extra kwargs
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction


class SimpleVQVAE(nn.Module):
    """
    Simplified VQVAE-like model for testing.
    Returns (reconstruction, additional_outputs) tuple like real VQVAE.
    """
    def __init__(
        self,
        in_channels: int = 10,
        hidden_dims: Optional[list] = None,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        **kwargs  # Accept and ignore extra kwargs
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        
        if hidden_dims is None:
            hidden_dims = [32, 64]
        
        # Encoder
        encoder_layers = []
        prev_dim = in_channels
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Quantization (simplified - just a linear layer)
        self.quantize = nn.Linear(embedding_dim, embedding_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = embedding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, in_channels))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def quantize_embeddings(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Simplified quantization"""
        z_q = self.quantize(z)
        # Return quantized embeddings and auxiliary info
        return z_q, {
            'commitment_loss': torch.tensor(0.0),
            'codebook_loss': torch.tensor(0.0),
        }
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass returning (reconstruction, aux_outputs) tuple.
        This mimics real VQVAE behavior for testing.
        """
        # Encode
        z = self.encode(x)
        
        # Quantize
        z_q, quant_info = self.quantize_embeddings(z)
        
        # Decode
        reconstruction = self.decode(z_q)
        
        # Return tuple: (reconstruction, auxiliary_outputs)
        aux_outputs = {
            'commitment_loss': quant_info['commitment_loss'],
            'codebook_loss': quant_info['codebook_loss'],
        }
        
        return reconstruction, z_q, aux_outputs


class TinyMLPClassifier(nn.Module):
    """
    Tiny MLP for quick testing.
    Single hidden layer.
    """
    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 10,
        hidden_dim: int = 32,
        embedding_dim: int = 16,  # Compatibility param
        **kwargs
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DummyConvModel(nn.Module):
    """
    Dummy convolutional model for testing with 2D/3D data.
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embedding_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, out_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, seq_len)
        return self.conv(x)


# ===== Model Registry for Tests =====

TEST_MODEL_MAP = {
    'simple_ae': SimpleAutoencoder,
    'simple_vqvae': SimpleVQVAE,
    'tiny_mlp': TinyMLPClassifier,
    'dummy_conv': DummyConvModel,
}


def get_test_model(model_type: str = 'simple_vqvae', **kwargs):
    """
    Factory function to create test models.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated model
    
    Example:
        >>> model = get_test_model('simple_vqvae', in_channels=10, embedding_dim=16)
    """
    if model_type not in TEST_MODEL_MAP:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(TEST_MODEL_MAP.keys())}")
    
    model_class = TEST_MODEL_MAP[model_type]
    return model_class(**kwargs)


# ===== Pytest Fixtures =====

def pytest_configure(config):
    """Register test models as pytest fixtures"""
    pass


def simple_autoencoder_factory(**kwargs):
    """Factory for SimpleAutoencoder"""
    return SimpleAutoencoder(**kwargs)


def simple_vqvae_factory(**kwargs):
    """Factory for SimpleVQVAE"""
    return SimpleVQVAE(**kwargs)