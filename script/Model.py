import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Block
        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for SE module.
        """
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3))  # Global Average Pooling
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        """
        Bottleneck Block with Residual Connection
        Args:
            in_channels (int): Number of input channels.
            mid_channels (int): Number of intermediate channels for bottleneck.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolution layers.
        """
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels,
                               kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return torch.relu(out)


class ResNet18(nn.Module):
    H = 224
    W = 224

    def __init__(self, model=None, bins=2, w=0.4, batch_size=7):
        """
        Enhanced ResNet-based architecture with Bottleneck and SE blocks, and residual connections.
        Args:
            model (torchvision.models): Pretrained ResNet model.
            bins (int): Number of bins for orientation classification.
            w (float): Weight parameter for orientation loss.
        """
        super(ResNet18, self).__init__()
        self.bins = bins
        self.w = w
        self.batch_size = batch_size

        # Feature extractor with SE Block and Bottleneck Residual connections
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SEBlock(64),

            BottleneckBlock(64, 64, 256, stride=2),
            BottleneckBlock(256, 64, 256),
            BottleneckBlock(256, 128, 512, stride=2),
            # Keep remaining layers of the pre-trained ResNet
            *list(model.children())[5:-2]
        )

        # Dimension head
        self.dimension = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 3)  # x, y, z
        )

        # Orientation head
        self.orientation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, bins * 2),  # 2 * bins for cosine and sine values
        )

        # Confidence head
        self.confidence = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, bins),  # Confidence scores for each bin
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        # Check the shape after feature extraction
        assert x.shape == (self.batch_size, 512, self.H,
                           self.W), f"Unexpected shape: {x.shape}"

        # Dimension prediction
        dimension = self.dimension(x)
        assert dimension.shape == (
            x.size(0), 3), f"Unexpected shape for dimension: {dimension.shape}"

        # Flatten for orientation and confidence heads
        x_flat = x.view(x.size(0), -1)

        # Orientation prediction
        orientation = self.orientation(x_flat)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)

        assert orientation.shape == (x.size(
            0), self.bins, 2), f"Unexpected shape for orientation: {orientation.shape}"

        # Confidence prediction
        confidence = self.confidence(x_flat)
        assert confidence.shape == (
            x.size(0), self.bins), f"Unexpected shape for confidence: {confidence.shape}"

        return orientation, confidence, dimension


if __name__ == '__main__':
    print("test")
