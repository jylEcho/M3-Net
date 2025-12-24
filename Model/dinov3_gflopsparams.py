# model/Dinov3B.py
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as F

# 可选依赖：timm（优先使用它来创建 ViT backbone）
try:
    from timm import create_model
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

class Dinov3_3DAdapter(nn.Module):
    def __init__(self,
                 num_classes=2,
                 input_channels=1,
                 target_size=(224,224),
                 backbone_name='vit_base_patch16_224',
                 weight_path='/root/autodl-tmp/dinov3-b/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'):
        """
        backbone_name: timm model name to try if timm available.
        weight_path: 本地权重文件（你给的路径）
        """
        super().__init__()
        self.input_channels = input_channels
        self.target_size = target_size
        self.weight_path = weight_path

        # channel adapter if necessary
        if input_channels != 3:
            self.channel_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)

        self.resize = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)

        # attempt to build backbone
        self.backbone = None
        self.backbone_type = None
        load_errors = []

        # 1) try timm backbone if timm installed
        if _HAS_TIMM:
            try:
                model = create_model(backbone_name, pretrained=False, num_classes=0)  # num_classes=0 -> feature model
                self.backbone = model
                self.backbone_type = 'timm'
            except Exception as e:
                load_errors.append(f"timm create_model failed: {e}")

        # 2) fallback: try to load as a vanilla ViT-like using torch.nn.Module wrappers
        if self.backbone is None:
            try:
                # Try loading with torch.load and inspect keys later; we'll keep a placeholder minimal conv backbone
                self.backbone = None
                self.backbone_type = 'state_dict_only'
            except Exception as e:
                load_errors.append(f"fallback construct failed: {e}")

        # classification head (will be created after we know hidden dim)
        # We'll set a default hidden_dim=768 (ViT-B) and create classifier; if backbone differs user can replace.
        hidden_dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, num_classes)
        )

        # attempt to load weights now (robust loader)
        if os.path.exists(self.weight_path):
            self._robust_load_weights(self.weight_path)
        else:
            print(f"[Dinov3_3DAdapter] weight file not found at {self.weight_path} (will train from scratch).")

    def _strip_module(self, state_dict):
        # remove "module." prefix if present
        new_sd = {}
        for k,v in state_dict.items():
            new_k = k
            if k.startswith('module.'):
                new_k = k[len('module.'):]
            new_sd[new_k] = v
        return new_sd

    def _extract_candidate_subdict(self, state_dict):
        # try to extract common sub-dicts like 'student', 'teacher', 'model', 'backbone'
        candidates = ['student', 'teacher', 'model', 'backbone', 'net']
        for c in candidates:
            if any(k.startswith(c + '.') for k in state_dict.keys()):
                return {k[len(c)+1:]: v for k,v in state_dict.items() if k.startswith(c + '.')}
        return state_dict

    def _robust_load_weights(self, path):
        print(f"[Dinov3_3DAdapter] trying to load weights from {path} ...")
        sd = torch.load(path, map_location='cpu')

        # if file is a dict with keys 'state_dict' or 'model'
        if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
            if 'state_dict' in sd:
                sd = sd['state_dict']
            elif 'model' in sd:
                sd = sd['model']

        # if the object itself isn't a dict (e.g. saved model object), bail
        if not isinstance(sd, dict):
            print("[Dinov3_3DAdapter] loaded object is not a state_dict/dict; aborting auto-load.")
            return

        # common cleanups
        sd = self._strip_module(sd)
        sd = self._extract_candidate_subdict(sd)

        # print a short diagnostic (first 30 keys)
        keys = list(sd.keys())
        print("[Dinov3_3DAdapter] state_dict keys sample (up to 30):")
        for k in keys[:30]:
            print("   ", k)
        if len(keys) > 30:
            print("   ... (total keys: {})".format(len(keys)))

        # If we have a timm backbone, try to load matching keys
        if self.backbone is not None and self.backbone_type == 'timm':
            # try matching by name: timm backbone.state_dict keys usually like 'patch_embed.proj.weight', etc.
            try:
                # some checkpoints may have a top-level 'head' we don't need; load strict=False
                self.backbone.load_state_dict(sd, strict=False)
                print("[Dinov3_3DAdapter] loaded weights into timm backbone (strict=False).")
                # try to infer hidden_dim from backbone
                try:
                    # many timm ViT have attribute embed_dim or num_features
                    hidden_dim = getattr(self.backbone, 'embed_dim', None) or getattr(self.backbone, 'num_features', None)
                    if hidden_dim is None and hasattr(self.backbone, 'head'):  # some models expose head.in_features
                        head = getattr(self.backbone, 'head')
                        if hasattr(head, 'in_features'):
                            hidden_dim = head.in_features
                    if hidden_dim:
                        # replace classifier to match hidden dim
                        self.classifier = nn.Sequential(
                            nn.Linear(hidden_dim, max(hidden_dim//2, 32)),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(max(hidden_dim//2, 32), self.classifier[-1].out_features)
                        )
                except Exception:
                    pass
                return
            except Exception as e:
                print("[Dinov3_3DAdapter] timm load_state_dict failed:", e)

        # As fallback attempt to map common naming patterns -> try to pick any vector parameters compatible with a ViT-like backbone
        # We'll try to locate "patch_embed.proj.weight" or "patch_embed.proj" etc.
        possible_match_keys = [k for k in sd.keys() if 'patch_embed' in k or 'pos_embed' in k or 'cls_token' in k]
        if possible_match_keys:
            print("[Dinov3_3DAdapter] found ViT-like keys in checkpoint (patch_embed/pos_embed/cls_token). Attempting to load into a timm-created vit model.")
            if not _HAS_TIMM:
                print("[Dinov3_3DAdapter] timm not installed; cannot create vit backbone automatically. Install timm and try again: pip install timm")
                return
            try:
                # recreate a standard ViT and load
                model = create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
                model.load_state_dict(sd, strict=False)
                self.backbone = model
                self.backbone_type = 'timm'
                print("[Dinov3_3DAdapter] fallback: created standard vit_base_patch16_224 and loaded weights (strict=False).")
                return
            except Exception as e:
                print("[Dinov3_3DAdapter] fallback create/load failed:", e)

        print("[Dinov3_3DAdapter] 自动加载未能完全匹配权重。你可以执行以下操作：")
        print("  1) 查看上面打印的键名样例，告诉我你看到的键名，我帮你写映射。")
        print("  2) 如果 checkpoint 包含子字典（例如 'student'/'teacher'），请手动指定，我会提取对应部分。")
        print("  3) 如果你使用 custom DINOv3 repo，请提供 backbone 构造代码文件路径。")

    def _get_cls_from_backbone(self, x):
        """
        Try several ways to extract CLS features from backbone:
        - if timm ViT: use forward_features() -> may return (B, num_tokens, dim) or (B, dim)
        - otherwise, if backbone returns features directly, accept that
        """
        if self.backbone is None:
            raise RuntimeError("backbone is None — 权重未加载且未建立backbone。查看初始化或权重文件。")

        # timm ViT usually has forward_features that returns either (B, num_tokens, dim) or (B, dim)
        if self.backbone_type == 'timm' and hasattr(self.backbone, 'forward_features'):
            feats = self.backbone.forward_features(x)
            if feats is None:
                # try forward
                feats = self.backbone(x)
            if feats.ndim == 3:
                # take cls token
                return feats[:, 0, :]  # (B, dim)
            elif feats.ndim == 2:
                return feats  # already (B, dim)
            else:
                raise RuntimeError(f"unexpected feats ndim: {feats.ndim}")
        else:
            # generic try
            out = self.backbone(x)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                return out[:,0,:]
            elif out.ndim == 2:
                return out
            else:
                raise RuntimeError("无法从backbone输出中获取cls token。")

    def forward(self, x):
        """
        x: (B, C, H, W, D)  - note: your earlier code used shape (B, C, H, W, D)
        We'll follow same contract.
        """
        x = x.unsqueeze(1)
        # ensure input shape consistent: if user provided (B, C, H, W, D), no change
        if x.ndim != 5:
            raise ValueError("Expect input with 5 dims (B, C, H, W, D). Got: {}".format(x.shape))
        B, C, H, W, D = x.shape

        # merge B and D
        x = x.permute(0, 4, 1, 2, 3)  # -> (B, D, C, H, W)
        x = x.reshape(-1, C, H, W)    # -> (B*D, C, H, W)

        # resize to target
        x = self.resize(x)  # (B*D, C, 224, 224)

        # channel adapter
        if C != 3:
            x = self.channel_adapter(x)

        # get CLS features
        cls = self._get_cls_from_backbone(x)  # expected (B*D, dim)
        # restore B and aggregate across depth
        cls = cls.reshape(B, D, -1)
        aggregated = cls.mean(dim=1)
        # classification
        out = self.classifier(aggregated)
        return out
# test_dinov3.py
# import torch
# from model.Dinov3B import Dinov3_3DAdapter

# if __name__ == "__main__":
#     # ===============================
#     # 保持你要求的输入格式不变
#     # x: (B, 1, 32, 32, 32)
#     # ===============================
#     x = torch.randn(1, 32, 32, 32).cuda()

#     print("Input shape:", x.shape)

#     # ===============================
#     # 创建模型（与脚本保持一致）
#     # ===============================
#     model = Dinov3_3DAdapter(
#         num_classes=2,
#         input_channels=1,
#         target_size=(224, 224),
#         backbone_name='vit_base_patch16_224',
#         weight_path='/root/autodl-tmp/dinov3-b/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
#     ).cuda()

#     model.eval()

#     # ===============================
#     # 前向传播
#     # ===============================
#     with torch.no_grad():
#         out = model(x)

#     print("Output shape:", out.shape)
#     print("Output:", out)
    
    
    
    
if __name__ == "__main__":
    import torch
    from thop import profile, clever_format

    # ===== 输入格式保持一致：(B, C, H, W, D) =====
    x = torch.randn(1, 32, 32, 32).cuda()

    print(f"Input shape: {x.shape}")

    # ===== 创建模型 =====
    model = Dinov3_3DAdapter(
        num_classes=2,
        input_channels=1,
        target_size=(224, 224),
        backbone_name='vit_base_patch16_224',
        weight_path='/root/autodl-tmp/dinov3-b/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    ).cuda()


    # ===== 前向传播 =====
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")

    # ===== 计算 FLOPs 和 Params =====
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print("\n========= Model Complexity =========")
    print(f"Params: {params}")
    print(f"FLOPs:  {flops}")
    print("====================================\n")
