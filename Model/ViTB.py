import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import torchvision.transforms.functional as F

class ViT3DAdapter(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, target_size=(224, 224)):
        super().__init__()
        # 本地ViT-B权重路径
        self.local_model_path = "/root/autodl-tmp/Foundation_model/Vit-B"
        
        # 从本地加载预训练模型和配置
        self.vit = ViTModel.from_pretrained(self.local_model_path)
        self.vit_config = ViTConfig.from_pretrained(self.local_model_path)
        
        # 适配输入通道（ViT默认3通道，处理单通道医学图像）
        if input_channels != 3:
            self.channel_adapter = nn.Conv2d(
                input_channels, 3, kernel_size=1, stride=1, padding=0
            )
        self.resize = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
        # ViT隐藏层维度（Vit-B通常为768）
        vit_hidden_dim = self.vit_config.hidden_size
        
        # 分类头：聚合深度特征后输出2类
        self.classifier = nn.Sequential(
            nn.Linear(vit_hidden_dim, vit_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(vit_hidden_dim // 2, num_classes)
        )

#     def forward(self, x):
#         # 输入形状：(B, C, H, W, D) （BCHWD）
#         # print(x.shape)
#         x = x.unsqueeze(1)
#         # print(x.shape)
#         # B, C, D, H, W = x.shape
#         # print(B)
#         # print(C)
#         # print(D)
#         # print(H)
#         # print(W)
#         B, C, D, H, W = x.shape
      
#         # 1. 合并B和D维度：(B, C, H, W, D) → (B×D, C, H, W)
#         x = x.permute(0, 4, 1, 2, 3)  # 变为 (B, D, C, H, W)
#         x = x.reshape(-1, C, H, W)    # 合并为 (B×D, C, H, W)
        
#         # 2. 适配ViT的3通道输入（单通道→3通道）
#         if C != 3:
#             x = self.channel_adapter(x)  # (B×D, 3, H, W)
        
#         # 3. 用本地加载的ViT提取特征（取[CLS] token）
#         vit_outputs = self.vit(pixel_values=x)
#         cls_features = vit_outputs.last_hidden_state[:, 0, :]  # (B×D, 768)
        
#         # 4. 还原B维度并聚合D个深度特征（均值池化）
#         cls_features = cls_features.reshape(B, D, -1)  # (B, D, 768)
#         aggregated = cls_features.mean(dim=1)  # (B, 768)
        
#         # 5. 输出分类结果（B, 2）
#         out = self.classifier(aggregated)
#         return out
    def forward(self, x):
        x = x.unsqueeze(1)
        # 输入形状：(B, C, H, W, D) （BCHWD），其中H=32, W=32
        B, C, D, H, W = x.shape
        # print(B)
        # print(C)
        # print(D)
        # print(H)
        # print(W)
        # 1. 合并B和D维度：(B, C, H, W, D) → (B×D, C, H, W)
        x = x.permute(0, 4, 1, 2, 3)  # 变为 (B, D, C, H, W)
        x = x.reshape(-1, C, H, W)    # 合并为 (B×D, C, H, W)
        
        # 2. 将32x32调整为224x224
        x = self.resize(x)  # 现在形状为 (B×D, C, 224, 224)
        
        # 3. 适配ViT的3通道输入（单通道→3通道）
        if C != 3:
            x = self.channel_adapter(x)  # (B×D, 3, 224, 224)
        
        # 4. 用本地加载的ViT提取特征（取[CLS] token）
        vit_outputs = self.vit(pixel_values=x)
        cls_features = vit_outputs.last_hidden_state[:, 0, :]  # (B×D, 768)
        
        # 5. 还原B维度并聚合D个深度特征（均值池化）
        cls_features = cls_features.reshape(B, D, -1)  # (B, D, 768)
        aggregated = cls_features.mean(dim=1)  # (B, 768)
        
        # 6. 输出分类结果（B, 2）
        out = self.classifier(aggregated)
        return out

# 测试代码
if __name__ == "__main__":
    # 输入示例：(B=2, C=1, H=224, W=224, D=5) （肺结节3D数据）
    x = torch.randn(2, 1, 224, 224, 5)
    
    # 初始化模型（从本地加载权重）
    model = ViT3DAdapter(num_classes=2, input_channels=1)
    
    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 (2, 2)