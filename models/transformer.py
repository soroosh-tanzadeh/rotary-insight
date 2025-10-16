import torch
from torch import nn
from .embedding import EmbeddingLinear


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        with_class_token=True,
        feedforward_dim=256,
        num_heads=8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.with_class_token = with_class_token

        self.layer_norm0 = nn.LayerNorm(embedding_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.LeakyReLU(),
            nn.Linear(feedforward_dim, embedding_dim),
        )

    def forward(self, x):
        x = self.layer_norm0(x)
        x = x + self.attention(x, x, x)[0]

        x = self.layer_norm1(x)

        x_feedforward = self.feedforward(x)
        x = x + x_feedforward

        x = self.layer_norm2(x)
        return x

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        patch_size=32,
        embedding_dim=64,
        positional_encoding=True,
        with_class_token=True,
        feedforward_dim=256,
        num_heads=8,
        num_encoders=12,
        number_of_channels=2,
        num_classes=10,
        dropout_rate=0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.with_class_token = with_class_token

        self.embedding = EmbeddingLinear(
            positional_encoding=positional_encoding,
            embedding_dim=embedding_dim,
            input_dim=input_dim,
            patch_size=patch_size,
            with_class_token=with_class_token,
            number_of_channels=number_of_channels,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.num_encoders = num_encoders
        for i in range(num_encoders):
            setattr(
                self,
                f"encoder{i}",
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    with_class_token=with_class_token,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                ),
            )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, num_classes),
        )

    def encode(self, x):
        x = self.embedding(x)
        for i in range(self.num_encoders):
            x = getattr(self, f"encoder{i}")(x)
            x = self.dropout(x)

        if self.with_class_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return x

    def forward(self, x):
        x = self.embedding(x)
        for i in range(self.num_encoders):
            x = getattr(self, f"encoder{i}")(x)
            x = self.dropout(x)

        if self.with_class_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        x = self.classifier(x)
        return x

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    model = TransformerEncoderClassifier(
        input_dim=2048,
        patch_size=16,
        embedding_dim=64,
        positional_encoding=False,
        num_classes=10,
    )
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(32, 2, 2048)
    y = model(x)
    print(y.shape)
