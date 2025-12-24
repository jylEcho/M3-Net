import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim_q // num_heads) ** -0.5

        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        # q: [B, Nq, Dq], kv: [B, Nk, Dkv]
        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        B, Nq, D = Q.shape
        Q = Q.view(B, Nq, self.num_heads, D // self.num_heads).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, D // self.num_heads).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, D // self.num_heads).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)
        return out


class Model(nn.Module):
    def __init__(self, net32, net64, net96, freeze_backbones=True, num_classes=2):
        super().__init__()
        self.net32 = net32
        self.net64 = net64
        self.net96 = net96

        if freeze_backbones:
            for net in [self.net32, self.net64, self.net96]:
                for p in net.parameters():
                    p.requires_grad = False

        embed_dim = 128
        self.proj32 = nn.Linear(2, embed_dim)
        self.proj64 = nn.Linear(2, embed_dim)
        self.proj96 = nn.Linear(2, embed_dim)

        # Cross-attention modules
        self.cross1 = CrossAttentionBlock(embed_dim, embed_dim, num_heads=4)
        self.cross2 = CrossAttentionBlock(embed_dim, embed_dim, num_heads=4)

        # Hierarchical Transformer Fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, img32, img64, img96):
        f32 = self.proj32(self.net32(img32))
        f64 = self.proj64(self.net64(img64))
        f96 = self.proj96(self.net96(img96))

        # Step 1: fuse low-level features
        f_small = torch.stack([f32, f64], dim=1)
        f_small = self.cross1(f_small, f_small)

        # Step 2: cross-attend with high-level feature
        f96 = f96.unsqueeze(1)
        # f_cross = self.cross2(f96, f_small)

        # Step 3: final transformer fusion
        tokens = torch.cat([f_small, f96], dim=1)
        fused = self.transformer_fusion(tokens)

        # Global pooling
        fused_feat = fused.mean(dim=1)
        fused_feat = self.norm(fused_feat)

        out = self.fc_head(fused_feat)
        return out
