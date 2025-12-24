'''Dual Path Networks in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

debug = False  # True
class Bottleneck(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
        super(Bottleneck, self).__init__()
        self.out_planes = out_planes
        self.dense_depth = dense_depth
        self.last_planes = last_planes
        self.in_planes = in_planes

        self.conv1 = nn.Conv3d(last_planes, in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm3d(in_planes)
        self.conv3 = nn.Conv3d(in_planes, out_planes + dense_depth, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes + dense_depth)

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv3d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes + dense_depth)
            )

    def forward(self, x):
        # 修改print语句为Python 3的括号形式
        # print('bottleneck_0', x.size(), self.last_planes, self.in_planes, 1)
        out = F.relu(self.bn1(self.conv1(x)))
        # print('bottleneck_1', out.size(), self.in_planes, self.in_planes, 3)
        out = F.relu(self.bn2(self.conv2(out)))
        # print('bottleneck_2', out.size(), self.in_planes, self.out_planes+self.dense_depth, 1)
        out = self.bn3(self.conv3(out))
        # print('bottleneck_3', out.size())
        x = self.shortcut(x)
        d = self.out_planes
        # print('bottleneck_4', x.size(), self.last_planes, self.out_planes+self.dense_depth, d)
        out = torch.cat([x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1)
        # print('bottleneck_5', out.size())
        out = F.relu(out)
        return out


class DPN(nn.Module):
    def __init__(self, cfg):
        super(DPN, self).__init__()
        in_planes, out_planes = cfg['in_planes'], cfg['out_planes']
        num_blocks, dense_depth = cfg['num_blocks'], cfg['dense_depth']

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.last_planes = 64
        self.layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2)
        self.layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.linear = nn.Linear(out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 2)  # 10)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
            # 注释掉无法执行的打印语句（原代码有误）
            # print('_make_layer', i, layers[-1].size())
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("*****x_shape1*****:", x.shape)
        x = x.unsqueeze(1)
        # print("*****x_shape2*****:", x.shape)
        if debug:
            print('0', x.size(), 64)
        out = F.relu(self.bn1(self.conv1(x)))
        if debug:
            print('1', out.size())
        out = self.layer1(out)
        if debug:
            print('2', out.size())
        out = self.layer2(out)
        if debug:
            print('3', out.size())
        out = self.layer3(out)
        if debug:
            print('4', out.size())
        out = self.layer4(out)
        if debug:
            print('5', out.size())
        out = F.avg_pool3d(out, 4)
        if debug:
            print('6', out.size())
        out_1 = out.view(out.size(0), -1)
        if debug:
            print('7', out_1.size())
        out = self.linear(out_1)
        if debug:
            print('8', out.size())
        # print(out.shape)
        # print(out_1.shape)
        return out


def DPN26():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (2, 2, 2, 2),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)


def DPN92_3D():
    cfg = {
        'in_planes': (96, 192, 384, 768),
        'out_planes': (256, 512, 1024, 2048),
        'num_blocks': (3, 4, 20, 3),
        'dense_depth': (16, 32, 24, 128)
    }
    return DPN(cfg)


def test():
    global debug  # 使用global关键字声明修改全局变量
    debug = True
    net = DPN92_3D()
    # Python 3中Variable已可直接使用，无需额外处理
    x = Variable(torch.randn(1, 1, 32, 32, 32))
    y = net(x)
    print(y)

if __name__ == "__main__":
    import time
    import torch
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 构建模型（使用你文件内定义的 DPN92_3D）
    model = DPN92_3D()
    # 若想用 DataParallel（多卡），可以打开下面：
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()

    # 根据你的要求，传入 (1, 32, 32, 56)
    # 模型内部会做 x = x.unsqueeze(1) -> (1,1,32,32,56)
    dummy_input = torch.randn(1, 32, 32, 56, device=device)

    # 前向检查
    with torch.no_grad():
        out = model(dummy_input)
    print("Forward OK. Output shape:", out.shape)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.3f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.3f} M)")

    # 尝试使用 thop 来计算 MACs / FLOPs
    try:
        from thop import profile
        # thop.profile 要求 inputs=(...), 这里传入我们构造的 dummy_input
        macs, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
        flops = 2 * macs  # 常用近似：FLOPs = 2 * MACs
        gflops = flops / 1e9
        print(f"THOP: MACs: {int(macs):,}, FLOPs: {int(flops):,}, GFLOPS: {gflops:.4f}")
        print(f"THOP params: {int(params_thop):,} (may slightly differ from direct count)")
    except Exception as e:
        print("thop not available or failed. Install with: pip install thop")
        print("thop exception:", repr(e))

    # 简单延时基准（warmup + repeats）
    repeats = 20
    # warmup runs
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    avg_latency_ms = (t1 - t0) / repeats * 1000.0
    print(f"Average latency over {repeats} runs: {avg_latency_ms:.2f} ms per forward (batch size = 1)")

    # 如果你想测 batch>1，只需修改 dummy_input 的第0维为 batch 大小

