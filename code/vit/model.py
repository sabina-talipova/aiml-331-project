import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=3, num_classes=4,
                 embed_dim=60, num_heads=4, num_layers=4, use_positional=True):
        super().__init__()

        # Size of each patch
        self.patch_size = patch_size
        # Total number of patches per image
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_positional = use_positional

        # Linear projection of image patches (like embedding): B x C x H x W → B x embed_dim x H/p x W/p
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # Learnable class token: acts as a representation for the entire image
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Optional learnable positional embeddings for patch positions + class token
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim)) if use_positional else None

        # Transformer encoder layer: contains multi-head self-attention and feedforward network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,          # embedding dimension
            nhead=num_heads,            # number of attention heads
            dim_feedforward=128,        # size of the feedforward layer
            activation='gelu',          # GELU activation function
            batch_first=True            # input format: (batch, sequence, embedding)
        )

        # Stack multiple transformer encoder layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization before classification head
        self.norm = nn.LayerNorm(embed_dim)

        # Linear classifier head for final prediction
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]  # batch size

        # Step 1: Patch embedding
        # Input: B x 3 x 128 x 128 → B x embed_dim x 8 x 8 (for patch_size=16)
        x = self.patch_embed(x)

        # Step 2: Flatten and reshape to a sequence of patch embeddings
        # B x embed_dim x H/p x W/p → B x num_patches x embed_dim
        x = x.flatten(2).transpose(1, 2)

        # Step 3: Prepend the class token to the sequence
        cls_token = self.cls_token.expand(B, 1, self.embed_dim)  # B x 1 x embed_dim
        x = torch.cat((cls_token, x), dim=1)                     # B x (1 + num_patches) x embed_dim

        # Step 4: Add positional embeddings (if enabled)
        if self.use_positional:
            x = x + self.pos_embedding[:, :x.size(1), :]

        # Step 5: Pass through transformer encoder
        x = self.encoder(x)

        # Step 6: Use only the class token for classification (x[:, 0])
        x = self.norm(x[:, 0])

        # Step 7: Final classification
        return self.head(x)
