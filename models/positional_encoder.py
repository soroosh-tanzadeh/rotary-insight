import torch
from torch import nn


# Fixed PositionalEncoder
class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim, seq_len, n=10000):
        super().__init__()
        self.embedding_dim = torch.tensor(embedding_dim)
        self.seq_len = torch.tensor(seq_len)
        self.n = torch.tensor(n)

        # Precompute encoding matrix during initialization
        encoding_matrix = torch.zeros(seq_len, embedding_dim)
        for pos in range(seq_len):
            for i in range(embedding_dim // 2):
                denominator = torch.pow(n, torch.tensor(2 * i) / embedding_dim)
                encoding_matrix[pos, 2 * i] = torch.sin(pos / denominator)
                encoding_matrix[pos, 2 * i + 1] = torch.cos(pos / denominator)

        # Register as buffer to move with device
        self.register_buffer(
            "encoding_matrix", encoding_matrix.unsqueeze(0)
        )  # [1, seq_len, dim]

    def forward(self, x):
        # x shape: [batch, seq_len, dim] or [seq_len, dim]
        if x.dim() == 2:
            return x + self.encoding_matrix.squeeze(0)
        else:
            return x + self.encoding_matrix


if __name__ == "__main__":
    embedding_dim = 4
    pe = PositionalEncoder(embedding_dim, 10)
    # Test
    print(pe(torch.randn((10, embedding_dim)), 0))
