import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.net_sphere import *

###################################################################################
###################################################################################
###################################################################################
'''
https://github.com/clcarwin/sphereface_pytorch/blob/master/net_sphere.py
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import dill
import math


def myphi(x, m):
    x = x * m
    return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
           x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)


import math
import torch
from torch import nn
# from scipy.special import binom
import scipy.special as special

class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin=4, device='cuda'):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        # self.device = device  # gpu or cpu
        use_cuda = not False and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(special.binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        a = 0
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        # self.mlambda = [
        #     lambda x: x ** 0,
        #     lambda x: x ** 1,
        #     lambda x: 2 * x ** 2 - 1,
        #     lambda x: 4 * x ** 3 - 3 * x,
        #     lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
        #     lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        # ]

    def forward(self, input):
        x = input  # size=(B,F)    F is feature len  (128*512)
        w = self.weight  # size=(F,Classnum) F=in_features Classnum=out_features
        # w = 512*227
        ww = w.renorm(2, 1, 1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5)  # size=B
        wlen = ww.pow(2).sum(0).pow(0.5)  # size=Classnum

        cos_theta = x.mm(ww)  # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1, 1) / wlen.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = 8 * cos_theta ** 4 - 8 * cos_theta ** 2 + 1
            theta = Variable(cos_theta.data.acos())
            k = (self.m * theta / 3.14159265).floor()
            n_one = k * 0.0 - 1
            phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta, self.m)
            phi_theta = phi_theta.clamp(-1 * self.m, 1)

        cos_theta = cos_theta * xlen.view(-1, 1)
        phi_theta = phi_theta * xlen.view(-1, 1)
        output = (cos_theta, phi_theta)
        return output  # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 1
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B, Classnum)
        # index = index.scatter(1, target.data.view(-1, 1).long(), 1)
        index = index.byte()
        index = Variable(index)
        # index = Variable(torch.randn(1,2)).byte()

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output1 = output.clone()
        # output1[index1] = output[index] - cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        # output1[index1] = output[index] + phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] = output1[index]- cos_theta[index] * (1.0 + 0) / (1 + self.lamb)+ phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        logpt = F.log_softmax(output)
        logpt = logpt.gather(1, target.long())
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.mean()
        # loss = torch.sum(cos_theta)+ torch.sum(phi_theta)
        return loss


# class STN(nn.Module):
#     def __init__(self ):
#         super(STN, self).__init__()
#         self.localization = nn.Sequential(
#             nn.Conv2d(3, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10*24*20, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         # self.fc_loc[2].weight.data.zero_()
#         # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#     def forward(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 10*24*20)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         return x


class sphere20a(nn.Module):
    def __init__(self, classnum=10574, feature=False):
        # classnum = dataloader.dataset.class_num = 227
        super(sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        # input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3, 64, 3, 2, 1)  # =>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128, 128, 3, 1, 1)  # =>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 6, 512)
        self.fc6 = AngleLinear(512, self.classnum)
        # self.stn = STN()

    def forward(self, x, target=None):
        # x = self.stn(x)
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0), -1)
        x = self.fc5(x)  # 128*512
        if self.feature:  # self.feature=False
            return x

        x = self.fc6(x)

        return x
###################################################################################
###################################################################################
###################################################################################

class testsp(nn.Module):
    def __init__(self):
        super(testsp, self).__init__()
        self.linear = AngleLinear(100, 2)

    def forward(self, x):
        out = self.linear(x)
        return out

def test():
    net = testsp()
    x = Variable(torch.randn(1,100))
    tar = Variable(torch.Tensor(1))
    out = net(x)
    cre = AngleLoss()
    loss = cre(out, tar)
    loss.backward()

debug = False


class ResCBAMLayer(nn.Module):
    def __init__(self, in_planes, feature_size):
        super(ResCBAMLayer, self).__init__()
        self.in_planes = in_planes
        self.feature_size = feature_size
        # self.ch_AvgPool = nn.AvgPool3d(feature_size, feature_size)
        # self.ch_MaxPool = nn.MaxPool3d(feature_size, feature_size)
        self.ch_AvgPool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 替换固定核大小的AvgPool3d
        self.ch_MaxPool = nn.AdaptiveMaxPool3d((1, 1, 1))  # 替换固定核大小的MaxPool3d        
        self.ch_Linear1 = nn.Linear(in_planes, in_planes // 4, bias=False)
        self.ch_Linear2 = nn.Linear(in_planes // 4, in_planes, bias=False)
        self.ch_Softmax = nn.Softmax(1)
        self.sp_Conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sp_Softmax = nn.Softmax(1)
        self.sp_sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_ch_avg_pool = self.ch_AvgPool(x).view(x.size(0), -1)
        x_ch_max_pool = self.ch_MaxPool(x).view(x.size(0), -1)
        # x_ch_avg_linear = self.ch_Linear2(self.ch_Linear1(x_ch_avg_pool))
        # print("x_ch_avg_pool_shape:", x_ch_avg_pool.shape)
        a = self.ch_Linear1(x_ch_avg_pool)
        x_ch_avg_linear = self.ch_Linear2(a)

        x_ch_max_linear = self.ch_Linear2(self.ch_Linear1(x_ch_max_pool))
        ch_out = (self.ch_Softmax(x_ch_avg_linear + x_ch_max_linear).view(x.size(0), self.in_planes, 1, 1, 1)) * x
        x_sp_max_pool = torch.max(ch_out, 1, keepdim=True)[0]
        x_sp_avg_pool = torch.sum(ch_out, 1, keepdim=True) / self.in_planes
        sp_conv1 = torch.cat([x_sp_max_pool, x_sp_avg_pool], dim=1)
        sp_out = self.sp_Conv(sp_conv1)
        sp_out = self.sp_sigmoid(sp_out.view(x.size(0), -1)).view(x.size(0), 1, x.size(2), x.size(3), x.size(4))
        out = sp_out * x + x
        return out


def make_conv3d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int,
                padding: int, dilation=1, groups=1,
                bias=True) -> nn.Module:
    """
    produce a Conv3D with Batch Normalization and ReLU

    :param in_channels: num of in in
    :param out_channels: num of out channels
    :param kernel_size: size of kernel int or tuple
    :param stride: num of stride
    :param padding: num of padding
    :param bias: bias
    :param groups: groups
    :param dilation: dilation
    :return: my conv3d module
    """
    module = nn.Sequential(

        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=groups,
                  bias=bias),
        nn.BatchNorm3d(out_channels),
        nn.ReLU())
    return module


def conv3d_same_size(in_channels, out_channels, kernel_size, stride=1,
                     dilation=1, groups=1,
                     bias=True):
    padding = kernel_size // 2
    return make_conv3d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


def conv3d_pooling(in_channels, kernel_size, stride=1,
                   dilation=1, groups=1,
                   bias=False):
    padding = kernel_size // 2
    return make_conv3d(in_channels, in_channels, kernel_size, stride,
                       padding, dilation, groups,
                       bias)


class ResidualBlock(nn.Module):
    """
    a simple residual block
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.my_conv1 = make_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.my_conv2 = make_conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = make_conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):
        out1 = self.conv3(inputs)
        out = self.my_conv1(inputs)
        out = self.my_conv2(out)
        out = out + out1
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1 = conv3d_same_size(in_channels=1, out_channels=4, kernel_size=3)
        self.conv2 = conv3d_same_size(in_channels=4, out_channels=4, kernel_size=3)
        self.config = config
        self.last_channel = 4
        self.first_cbam = ResCBAMLayer(4, 32)
        layers = []
        i = 0
        for stage in config:
            i = i+1
            layers.append(conv3d_pooling(self.last_channel, kernel_size=3, stride=2))
            for channel in stage:
                layers.append(ResidualBlock(self.last_channel, channel))
                self.last_channel = channel
            layers.append(ResCBAMLayer(self.last_channel, 32//(2**i)))
        self.layers = nn.Sequential(*layers)
        self.avg_pooling = nn.AvgPool3d(kernel_size=4, stride=4)
        # self.fc = AngleLinear(in_features=2048, out_features=2)
        self.fc = AngleLinear(in_features=512, out_features=2)

    def forward(self, inputs):
        # print("input_shape0:",inputs.shape)
        inputs = inputs.unsqueeze(1)
        # print("input_shape1:",inputs.shape)        
 
        if debug:
            print(inputs.size())
        # print("inputs_shape:", inputs.shape)
        out = self.conv1(inputs)
        if debug:
            print(out.size())
        out = self.conv2(out)
        if debug:
            print(out.size())
        # print("out_shape:",out.shape)
        out = self.first_cbam(out)
        out = self.layers(out)
        if debug:
            print(out.size())
        out = self.avg_pooling(out)
        out = out.view(out.size(0), -1)
        if debug:
            print(out.size())
        # print("out_shape1:",out.shape)   
        out = self.fc(out)
        # print("fc输出类型:", type(out))  # 确认是元组
        # print("元组长度:", len(out))
        # for i, item in enumerate(out):
        #     print(f"元组第{i}个元素形状:", item.shape)
        return out[0]


def test():
    global debug
    debug = True
    net = ConvRes([[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]])
    inputs = torch.randn((1, 1, 32, 32, 32))
    output = net(inputs)
    print(net.config)
    print(output)

    

if __name__ == "__main__":
    import torch
    from thop import profile, clever_format

    # ===== 模型配置 =====
    model_config = [[64, 64, 64],
                    [128, 128, 256],
                    [256, 256, 256, 512]]

    # ===== 实例化模型 =====
    model = Model(model_config).cuda()

    # ===== 构造输入 (B, C, D, H, W) =====
    x = torch.randn(1, 32, 32, 32).cuda()
    print(f"Input shape: {x.shape}")

    # ===== 前向传播 =====
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")

    # ===== 统计参数量与 FLOPs =====
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print("\n========= Model Complexity =========")
    print(f"Params: {params}")
    print(f"FLOPs:  {flops}")
    print("====================================\n")
