import torch
from torch import nn
from .positional_encoder import PositionalEncoder


def create_patches(x: torch.Tensor, patch_size=32):
    # Add batch dimension if needed
    if x.dim() == 2:
        x = x.unsqueeze(0)
        removed_batch_dim = True
    else:
        removed_batch_dim = False

    if x.shape[2] % patch_size != 0:
        raise RuntimeError(
            f"Input length {x.shape[2]} is not divisible by patch size {patch_size}"
        )

    # [batch, channels, num_patches, patch_size]
    patches = x.unfold(2, patch_size, patch_size)
    # [batch, num_patches, channels, patch_size]
    patches = patches.permute(0, 2, 1, 3)

    if removed_batch_dim:
        patches = patches.squeeze(0)
    return patches


class ClassToken(nn.Module):
    """Add learnable class token for classification"""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, num_patches, D)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls_tokens, x], dim=1)


class EmbeddingLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        patch_size=32,
        positional_encoding=False,
        with_class_token=False,
        number_of_channels=2,
        embedding_dim=32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.positional_encoding = positional_encoding
        self.with_class_token = with_class_token

        if self.positional_encoding:
            self.pe = PositionalEncoder(embedding_dim, self.num_patches)

        self.flatten = nn.Flatten(start_dim=1)
        self.cls_token = ClassToken(embed_dim=embedding_dim)

        self.encoder = nn.Linear(patch_size, embedding_dim // number_of_channels)

    def forward(self, x: torch.Tensor):
        ## What do?
        # 1. Create patches of input signals
        # 2. Combine patches and batch dim, to perform embedding for each patch separately
        # 3. Combine channels and perform embedding
        # 4. reshape back to [batch, num_patches, embedding_dim]. Transformer treat the the embedding as a sequence
        # 5. If positional encoding is active, add positional encoding
        # 6. If class token is active, add class token
        # 7. Return the output

        original_shape = x.shape
        # Add batch dim if needed
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Create patches: [batch, num_patches, channels, patch_size]
        patches = create_patches(x, self.patch_size)
        batch_size, num_patches, channels, _ = patches.shape

        # Combine batch and num_patches
        x = patches.reshape(batch_size * num_patches, channels, self.patch_size)

        x = self.encoder(x)

        x = x.view(batch_size, num_patches, -1)

        if self.positional_encoding:
            x = self.pe(x)

        # Remove batch dim if input didn't have it
        if len(original_shape) == 2:
            x = x.squeeze(0)

        # Add class token
        if self.with_class_token:
            x = self.cls_token(x)  # [batch, num_patches*2, embedding_dim]
        return x


if __name__ == "__main__":

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.x = torch.randn((32, 2, 2048))

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx]

    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
