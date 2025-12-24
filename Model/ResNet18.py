import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, input_channels=1):
        super(ResNet3D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7,
                               stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.conv1(x)     # (B, 64, D, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # (B, 64, D/2, H/4, W/4)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet3d18(num_classes=2, input_channels=1):
    # ✅ 这里调用类 ResNet3D，而不是 Model()
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes, input_channels)

# def resnet3d34(num_classes=2, input_channels=1):
    # return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes, input_channels)

# ✅ 统一的创建接口（供 main 调用）
def Model(model_name="resnet3d18", num_classes=2, input_channels=1):
    model_name = model_name.lower()
    if model_name == "resnet3d18":
        return resnet3d18(num_classes=num_classes, input_channels=input_channels)
    # if model_name == "resnet3d34":
        # return resnet3d34(num_classes=num_classes, input_channels=input_channels)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

# ✅ 测试
if __name__ == "__main__":
    model = Model(model_name="resnet3d18", num_classes=2, input_channels=1)
    x = torch.randn(1, 1, 56, 64, 64)
    print(model(x).shape)  # torch.Size([1, 2])