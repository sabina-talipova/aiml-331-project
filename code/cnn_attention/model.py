import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel Attention Module: focuses on "what" is important across the channel dimension
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        # MLP with bottleneck: reduces then restores channel dimension
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),  # Bottleneck layer
            nn.ReLU(),                                     # Non-linearity
            nn.Linear(in_channels // ratio, in_channels)   # Expand back to original channels
        )

        # Global average and max pooling (spatial -> channel descriptors)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Sigmoid function to scale attention weights between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool spatial dimensions to get channel descriptors
        avg = self.avg_pool(x).view(x.size(0), -1)
        max_ = self.max_pool(x).view(x.size(0), -1)

        # Pass through MLP
        avg_out = self.mlp(avg)
        max_out = self.mlp(max_)

        # Combine and reshape attention weights
        out = avg_out + max_out
        scale = self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

        # Apply channel-wise attention
        return x * scale


# Spatial Attention Module: focuses on "where" is important in the spatial dimensions
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        # Padding ensures output spatial size is same as input
        padding = kernel_size // 2

        # Convolution across concatenated average & max pooled feature maps
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compress channel dimension using average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate along the channel axis (shape: Bx2xHxW)
        concat = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution and sigmoid to get spatial attention map
        scale = self.sigmoid(self.conv(concat))

        # Apply spatial attention
        return x * scale


# Full CNN with integrated Channel and Spatial Attention (CBAM-style)
class AttentionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Convolutional block with both channel and spatial attention
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),      # Standard conv layer
                nn.BatchNorm2d(out_c),                     # Batch normalization
                nn.ReLU(),                                 # Non-linearity
                ChannelAttention(out_c),                   # Channel-wise attention
                SpatialAttention(),                        # Spatial attention
                nn.MaxPool2d(2)                            # Downsampling
            )

        # Feature extractor
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to reduce spatial dimension
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                # Flatten feature map to vector
            nn.Linear(128, 256),         # Fully connected layer
            nn.ReLU(),                   # Non-linearity
            nn.Linear(256, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.features(x)     # Extract features
        x = self.classifier(x)   # Classify
        return x
