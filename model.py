import torch
import torch.nn as nn
import torch.nn .functional as F
import torchvision.transforms.functional as TF

# Define DoubleConv
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(out_channel)
        )

    def forward(self, x):
        return self.conv(x)
    

# Define UNet
class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, features=[64, 128, 256, 512]):  # out is binary, so 1
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 161x161 -> 80x80 -> 160x160 concat problem

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            in_channel = feature
        
        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # bottleneck part of UNet
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # final conv(1x1)
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)
    
    # skip connection
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            # add down sample before pooling
            skip_connections.append(x)
            x = self.pool(x)
        
        # 512 -> 1024
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]   # reverse because it uses upsampling

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)    # 64, 128, ...
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
    
def test():
    x = torch.randn((3, 1, 160, 160))   # batch, channel, h, w
    model = UNet(in_channel=1, out_channel=1)
    preds = model(x)

    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
