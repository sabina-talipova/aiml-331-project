import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# SimpleCNN: A basic convolutional neural network for image classification.
# Designed for images of size 128x128 with RGB channels.
# ------------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4, use_batchnorm=True, activation=nn.ReLU):
        """
        Args:
            num_classes (int): Number of output classes.
            use_batchnorm (bool): Whether to include BatchNorm2d after each Conv2d.
            activation (nn.Module): Activation function to use (default: ReLU).
        """
        super().__init__()
        self.use_bn = use_batchnorm
        self.activation = activation()  # Instantiating the activation module

        # Internal helper function to create a conv block:
        # Conv2d -> (BatchNorm2d) -> Activation -> MaxPool2d
        def conv_block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)]
            if self.use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(self.activation)
            layers.append(nn.MaxPool2d(2))  # Halves the spatial dimensions
            return nn.Sequential(*layers)

        # Feature extractor part of the network:
        # Series of convolutional blocks that reduce the spatial dimensions and increase depth
        self.features = nn.Sequential(
            conv_block(3, 32),  # Input: [3 x 128 x 128] -> [32 x 64 x 64]
            conv_block(32, 64),  # -> [64 x 32 x 32]
            conv_block(64, 128),  # -> [128 x 16 x 16]
            conv_block(128, 256),  # -> [256 x 8 x 8]
            conv_block(256, 512),  # -> [512 x 4 x 4]
            nn.AdaptiveAvgPool2d((1, 1))  # -> [512 x 1 x 1]
        )

        # Dynamically compute the flattened dimension for the fully connected layer
        dummy_input = torch.zeros(1, 3, 128, 128)  # Simulated input image
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
            flattened_dim = dummy_output.view(1, -1).size(1)  # Typically 512

        # Fully connected classifier:
        # [B, 512] -> [B, 256] -> [B, num_classes]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------------------------
# ResidualBlock: A standard residual block with two convolutions and skip connection.
# ------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        # Two stacked convolutional layers with BatchNorm and ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # Skip connection (identity if channels match, else 1x1 convolution)
        self.skip = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass with residual connection.
        """
        out = self.conv(x)  # Main path
        x = self.skip(x)  # Skip connection path
        return self.relu(out + x)  # Combine paths


# ------------------------------------------------------------------------------
# ResidualCNN: A CNN using stacked residual blocks.
# Suitable for deeper networks with better gradient flow.
# ------------------------------------------------------------------------------
class ResidualCNN(nn.Module):
    def __init__(self, num_classes=4):
        """
        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()

        # Initial convolutional layer to prepare input for residual blocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Residual layers with increasing channel depth and spatial downsampling
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2)  # Downsample
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512),
            nn.MaxPool2d(2)
        )

        # Adaptive global pooling reduces [B, 512, H, W] to [B, 512, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [B, 512, 1, 1] -> [B, 512]
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through stem -> 4 residual layers -> global pool -> classifier
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

class ImprovedSimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),  # для 128x128 входа
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc(x)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return self.relu(out)

