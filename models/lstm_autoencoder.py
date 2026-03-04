from typing import Tuple

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time series anomaly detection.

    Expected input shape: (batch_size, sequence_length, num_features)
    """

    def __init__(
        self,
        num_features: int,
        latent_dim: int = 16,
        encoder_hidden_sizes: Tuple[int, int] = (64, 32),
        decoder_hidden_sizes: Tuple[int, int] = (32, 64),
    ) -> None:
        super().__init__()

        enc_h1, enc_h2 = encoder_hidden_sizes
        dec_h1, dec_h2 = decoder_hidden_sizes

        # Encoder: two stacked LSTM layers (64 units -> 32 units).
        self.encoder_lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=enc_h1,
            batch_first=True,
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=enc_h1,
            hidden_size=enc_h2,
            batch_first=True,
        )

        # Latent representation: project final encoder hidden state to 16-D.
        self.to_latent = nn.Linear(enc_h2, latent_dim)

        # Decoder: map latent vector to initial decoder feature size,
        # then repeat across the time dimension.
        self.from_latent = nn.Linear(latent_dim, dec_h1)

        # Decoder LSTMs: 16 (via dec_h1) -> 32 units -> 64 units.
        self.decoder_lstm1 = nn.LSTM(
            input_size=dec_h1,
            hidden_size=dec_h1,
            batch_first=True,
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=dec_h1,
            hidden_size=dec_h2,
            batch_first=True,
        )

        # Output layer: reconstruct original feature dimension at each timestep.
        self.output_layer = nn.Linear(dec_h2, num_features)

        # Reconstruction loss: Mean Squared Error.
        self.criterion = nn.MSELoss(reduction="mean")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequences into a latent representation.

        x: (batch_size, sequence_length, num_features)
        returns: (batch_size, latent_dim)
        """
        out, _ = self.encoder_lstm1(x)
        out, (h_n, _) = self.encoder_lstm2(out)

        # h_n from the last encoder LSTM layer has shape (num_layers, batch, hidden_size).
        h_last = h_n[-1]  # (batch, hidden_size)

        z = self.to_latent(h_last)  # (batch, latent_dim)
        return z

    def decode(self, z: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """
        Decode latent vectors back into sequences.

        z: (batch_size, latent_dim)
        returns: (batch_size, sequence_length, num_features)
        """
        # Map latent space back to decoder feature space.
        dec_input = self.from_latent(z)  # (batch, dec_h1)

        # "Repeat vector": replicate across the time dimension.
        dec_input = dec_input.unsqueeze(1).repeat(1, sequence_length, 1)

        out, _ = self.decoder_lstm1(dec_input)
        out, _ = self.decoder_lstm2(out)

        # Apply output layer at each timestep.
        batch_size, seq_len, hidden_dim = out.shape
        out_flat = out.reshape(batch_size * seq_len, hidden_dim)
        recon_flat = self.output_layer(out_flat)
        recon = recon_flat.view(batch_size, seq_len, -1)

        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        x: (batch_size, sequence_length, num_features)
        returns:
            recon: reconstructed sequences with same shape as x
            z:     latent representation
        """
        sequence_length = x.size(1)
        z = self.encode(x)
        recon = self.decode(z, sequence_length)
        return recon, z

    def reconstruction_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to compute MSE reconstruction loss.

        Returns:
            loss, recon, z
        """
        recon, z = self.forward(x)
        loss = self.criterion(recon, x)
        return loss, recon, z


if __name__ == "__main__":
    # Example usage with dummy data.
    batch_size = 8
    sequence_length = 10
    num_features = 12

    model = LSTMAutoencoder(num_features=num_features)

    # Random normalized input tensor.
    x = torch.rand(batch_size, sequence_length, num_features)

    recon, z = model(x)
    loss, _, _ = model.reconstruction_loss(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"MSE loss: {loss.item():.6f}")

