import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
#   Bottleneck Block for ResNet50
# -----------------------------
class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# -----------------------------
#   ResNet3D (General Framework)
# -----------------------------
class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, input_channels=1):
        super(ResNet3D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7,
                               stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # layers
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
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -----------------------------
#   ResNet50 3D
# -----------------------------
def resnet3d50(num_classes=2, input_channels=1):
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_classes, input_channels)


# -----------------------------
#   Unified model factory
# -----------------------------
def Model(model_name="resnet3d50", num_classes=2, input_channels=1):
    if model_name.lower() == "resnet3d50":
        return resnet3d50(num_classes=num_classes, input_channels=input_channels)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    import torch
    from thop import profile, clever_format

    # ======= 创建模型 =======
    model = Model(model_name="resnet3d50",
                  num_classes=2,
                  input_channels=1).cuda()

    # ======= 构造 4D 输入 (B, D, H, W) =======
    # 你的 forward 会自动 unsqueeze 成 (B, C=1, D, H, W)
    x = torch.randn(1, 32, 32, 32).cuda()
    print(f"Input shape: {x.shape}")

    # ======= 前向传播 =======
    with torch.no_grad():
        out = model(x)
        print(f"Output shape: {out.shape}")

    # ======= 计算 FLOPs 和 Params =======
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print("\n========= Model Complexity =========")
    print(f"Params: {params}")
    print(f"FLOPs:  {flops}")
    print("====================================\n")
