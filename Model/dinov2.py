import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config
import torchvision.transforms.functional as F

class DinoV2_3DAdapter(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, target_size=(224, 224)):
        super().__init__()

        # 本地 DINOv2-B 权重
        self.local_model_path = "/root/autodl-tmp/Foundation_model/dinov2-b"

        # 从本地加载 DINOv2 配置和模型
        self.dino = Dinov2Model.from_pretrained(self.local_model_path)
        self.dino_config = Dinov2Config.from_pretrained(self.local_model_path)

        # 适配单通道医学图像 → 3 通道
        if input_channels != 3:
            self.channel_adapter = nn.Conv2d(
                input_channels, 3, kernel_size=1, stride=1, padding=0
            )

        # 调整输入大小为 224×224
        self.resize = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)

        # DINOv2 hidden dim （base 模型通常 768）
        hidden_dim = self.dino_config.hidden_size

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )


    def forward(self, x):
        # 输入 (B, C, H, W, D)
        x = x.unsqueeze(1)  # → (B, 1, C, H, W) 你的 x 本来是 (B, H, W, D)
        B, C, D, H, W = x.shape

        # (B, C, H, W, D) → (B×D, C, H, W)
        x = x.permute(0, 4, 1, 2, 3)  
        x = x.reshape(-1, C, H, W)

        # 32×32 → 224×224 （如果原本是 224 就无影响）
        x = self.resize(x)

        # 单通道 → 3 通道
        if C != 3:
            x = self.channel_adapter(x)

        # DINOv2 提取 CLS 特征
        outputs = self.dino(pixel_values=x)
        cls_features = outputs.last_hidden_state[:, 0, :]  # (B×D, dim)

        # 合回 B 维，并对 D 聚合
        cls_features = cls_features.reshape(B, D, -1)
        aggregated = cls_features.mean(dim=1)

        # 分类
        out = self.classifier(aggregated)
        return out



# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 1, 224, 224, 5)  # (B, C, H, W, D)

    model = DinoV2_3DAdapter(num_classes=2, input_channels=1)
    out = model(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
