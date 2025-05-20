import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, image_size=128, patch_size=16, in_channels=3, num_classes=4,
                 embed_dim=60, num_heads=4, num_layers=4, use_positional=True):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.use_positional = use_positional

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim)) if use_positional else None

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=128,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)            # B x C x H x W
        x = x.flatten(2).transpose(1, 2)   # B x N x C

        cls_token = self.cls_token.expand(B, -1, -1)  # B x 1 x C
        x = torch.cat((cls_token, x), dim=1)          # B x (1+N) x C

        if self.use_positional:
            x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)
