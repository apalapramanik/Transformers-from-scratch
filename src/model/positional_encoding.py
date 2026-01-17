import torch
import math


class PositionalEncoding(torch.nn.Module):
    """
    PURPOSE:
    Inject word order information into embeddings.

    KEY IDEA:
    - Attention has no notion of sequence order
    - Positional encoding tells the model "where" each word is
    - Added directly to word embeddings

    TYPE:
    - Fixed (sinusoidal)
    - No learning parameters
    """

    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        # Create position indices: [0, 1, 2, ..., max_len]
        position = torch.arange(0, max_len).unsqueeze(1)

        # Compute scaling terms for different dimensions
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        # Initialize positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, embed_dim)
        # Batch dimension added for broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        INPUT:
        x : (batch, seq_len, embed_dim)

        OUTPUT:
        x + positional encoding
        """

        seq_len = x.size(1)

        # Add positional encoding (no parameters, no gradients)
        x = x + self.pe[:, :seq_len]

        return x
