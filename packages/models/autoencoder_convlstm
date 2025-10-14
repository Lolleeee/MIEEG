import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        """
        ConvLSTM cell implementation
        
        Args:
            input_dim: Number of channels of input tensor
            hidden_dim: Number of channels of hidden state
            kernel_size: Size of the convolutional kernel
            bias: Whether to add bias
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutional layer for all gates
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        # Update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        device = self.conv.weight.device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )


class ConvLSTMAE(nn.Module):
    def __init__(self, in_channels=25, latent_dim=128):
        """
        ConvLSTM Autoencoder with bilinear interpolation decoder
        Input shape: [batch, 50, 7, 5, 250]
        
        Args:
            in_channels: Number of input channels (50)
            latent_dim: Size of the latent embedding space
        """
        super(ConvLSTMAE, self).__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.spatial_size = (7, 5)
        
        # Encoder: Spatial downsampling + ConvLSTM
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.encoder_convlstm1 = ConvLSTMCell(input_dim=64, hidden_dim=64, kernel_size=3)
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.encoder_convlstm2 = ConvLSTMCell(input_dim=128, hidden_dim=128, kernel_size=3)
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.encoder_convlstm3 = ConvLSTMCell(input_dim=256, hidden_dim=256, kernel_size=3)
        
        # Calculate spatial dimensions after encoding
        # After stride=2 in encoder_conv2: 7->4, 5->3
        self.encoded_spatial = (4, 3)
        self.flat_size = 256 * 4 * 3
        
        # Bottleneck
        self.fc_encoder = nn.Linear(self.flat_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.flat_size)
        
        # Decoder: ConvLSTM + Bilinear upsampling
        self.decoder_convlstm1 = ConvLSTMCell(input_dim=256, hidden_dim=256, kernel_size=3)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Bilinear upsampling block
        self.decoder_convlstm2 = ConvLSTMCell(input_dim=256, hidden_dim=128, kernel_size=3)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.decoder_convlstm3 = ConvLSTMCell(input_dim=128, hidden_dim=64, kernel_size=3)
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def encode(self, x):
        """
        Encode spatiotemporal input to latent representation
        
        Args:
            x: [batch, channels, height, width, time]
        Returns:
            embedding: [batch, latent_dim]
        """
        batch_size = x.size(0)
        seq_len = x.size(4)
        
        # Initialize hidden states for each ConvLSTM layer
        h1, c1 = self.encoder_convlstm1.init_hidden(batch_size, self.spatial_size)
        h2, c2 = self.encoder_convlstm2.init_hidden(batch_size, (4, 3))
        h3, c3 = self.encoder_convlstm3.init_hidden(batch_size, (4, 3))
        
        # Process each timestep through encoder
        for t in range(seq_len):
            # Get frame at timestep t
            x_t = x[:, :, :, :, t]
            
            # Encoder layer 1
            x_t = self.encoder_conv1(x_t)
            h1, c1 = self.encoder_convlstm1(x_t, (h1, c1))
            
            # Encoder layer 2
            x_t = self.encoder_conv2(h1)
            h2, c2 = self.encoder_convlstm2(x_t, (h2, c2))
            
            # Encoder layer 3
            x_t = self.encoder_conv3(h2)
            h3, c3 = self.encoder_convlstm3(x_t, (h3, c3))
        
        # Use final hidden state as encoding
        encoded = h3.reshape(batch_size, -1)
        embedding = self.fc_encoder(encoded)
        
        return embedding
    
    def decode(self, embedding, seq_len=250):
        """
        Decode latent representation to spatiotemporal output using bilinear interpolation
        
        Args:
            embedding: [batch, latent_dim]
            seq_len: Number of timesteps to generate
        Returns:
            output: [batch, channels, height, width, time]
        """
        batch_size = embedding.size(0)
        
        # Project embedding back to spatial representation
        x = self.fc_decoder(embedding)
        x = x.reshape(batch_size, 256, 4, 3)
        
        # Initialize hidden states for decoder ConvLSTM layers
        h1, c1 = self.decoder_convlstm1.init_hidden(batch_size, (4, 3))
        h2, c2 = self.decoder_convlstm2.init_hidden(batch_size, self.spatial_size)
        h3, c3 = self.decoder_convlstm3.init_hidden(batch_size, self.spatial_size)
        
        outputs = []
        
        # Generate each timestep
        for t in range(seq_len):
            # Decoder layer 1
            h1, c1 = self.decoder_convlstm1(x, (h1, c1))
            x_t = self.decoder_conv1(h1)
            
            # Bilinear upsampling to (7, 5)
            x_t = F.interpolate(x_t, size=(7, 5), mode='bilinear', align_corners=False)
            
            # Decoder layer 2
            h2, c2 = self.decoder_convlstm2(x_t, (h2, c2))
            x_t = self.decoder_conv2(h2)
            
            # Decoder layer 3
            h3, c3 = self.decoder_convlstm3(x_t, (h3, c3))
            x_t = self.decoder_conv3(h3)
            
            outputs.append(x_t)
            
            # Use current hidden state as next input
            x = self.decoder_conv1(h1)
        
        # Stack all timesteps
        output = torch.stack(outputs, dim=4)
        return output
    
    def forward(self, x):
        """
        Full forward pass through autoencoder
        
        Args:
            x: [batch, channels, height, width, time]
        Returns:
            reconstruction: [batch, channels, height, width, time]
        """
        seq_len = x.size(4)
        embedding = self.encode(x)
        reconstruction = self.decode(embedding, seq_len)
        return reconstruction


# Example usage
if __name__ == "__main__":
    # Create model
    model = ConvLSTMAE(in_channels=50, latent_dim=128)
    print(model)

    # Test with sample input
    batch_size = 4
    x = torch.randn(batch_size, 50, 7, 5, 250)
    
    # Forward pass
    reconstruction = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstruction.shape}")
    
    # Get embedding
    embedding = model.encode(x)
    print(f"Embedding shape: {embedding.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
