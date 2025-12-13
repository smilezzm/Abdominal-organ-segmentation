import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=12, features=[64, 128, 256, 512], use_checkpointing=False):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_checkpointing = use_checkpointing

        # Downsampling Path (Encoder)
        for feature in features:
            self.downs.append(self._conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        # Upsampling Path (Decoder)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(self._conv_block(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # Encoder with gradient checkpointing
        for down in self.downs:
            if self.use_checkpointing and self.training:
                x = checkpoint(down, x, use_reentrant=False)
            else:
                x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck with checkpointing
        if self.use_checkpointing and self.training:
            x = checkpoint(self.bottleneck, x, use_reentrant=False)
        else:
            x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1] # Reverse the skip connections for the decoder path

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # If dimensions don't match (due to padding in conv or odd input sizes), resize skip connection
            # Assuming [batch, channel, height, width] for x and skip_connection
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Apply checkpointing to decoder blocks
            if self.use_checkpointing and self.training:
                x = checkpoint(self.ups[idx+1], concat_skip, use_reentrant=False)
            else:
                x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
