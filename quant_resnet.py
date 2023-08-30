import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import args_config
from torchvision import datasets, transforms
import gc
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from statistics import mean
from training_utils import Firing, w_q, b_q
from network_utils import *
from spike_related import LIFSpike

args = args_config.get_args()
# firing = Firing.apply


import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron

# tau_global = 1./(1. - 0.5)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, n_w, n_u, n_b, stride=1):
        super(BasicBlock, self).__init__()

        
        self.num_bits_w = n_w
        self.num_bits_b = n_b
        self.num_bits_u = n_u

        conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(planes)
        lif1 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif1 = QConvBN2dLIF(conv1,bn1,lif1,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(planes)
        self.ConvBn2 = QConvBN2d(conv2,bn2,self.num_bits_w,self.num_bits_u)


        self.lif2 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)

        self.shortcut = nn.Sequential()
        conv_sh = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        bn_sh = nn.BatchNorm2d(self.expansion*planes)
        # self.ConvBn_sh = QConvBN2d(conv_sh,bn_sh,self.num_bits_w,self.num_bits_u)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QConvBN2d(conv_sh,bn_sh,self.num_bits_w,self.num_bits_u,short_cut=True)
            )

    def forward(self, x):
        out = self.ConvBnLif1(x)
        out = self.ConvBn2(out)
        out += self.shortcut(x)
        out = self.lif2(out, args.share, self.ConvBn2.beta[0], bias=0)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, total_timestep =4):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.total_timestep = total_timestep
        if args.dataset == 'dvs':
            input_dim = 2
        else:
            input_dim = 3

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.num_bits_u = 16
        self.num_bits_w = 16
        self.num_bits_b = 8

        print("ResNet-basic-block weight bits: ", self.num_bits_w)
        print("ResNet-basic-block potential bits: ", self.num_bits_u)

        conv1dvs = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        bn1dvs = nn.BatchNorm2d(64,affine=True)
        lif1dvs =  LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif1 = QConvBN2dLIF(conv1dvs,bn1dvs,lif1dvs,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.direct_lif = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=False)
        # self.lif_input = neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
        #                          surrogate_function=surrogate.ATan(),
        #                          detach_reset=True)


        self.layer1 = self._make_layer(block, 128, num_blocks[0], self.num_bits_w, self.num_bits_u, self.num_bits_b, stride=2)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], self.num_bits_w, self.num_bits_u, self.num_bits_b, stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], self.num_bits_w, self.num_bits_u, self.num_bits_b, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512, 256)
        self.lif_fc = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=False)
        # self.lif_fc = neuron.ParametricLIFNode(v_threshold=1.0, v_reset=0.0, init_tau=2.,
        #                                           surrogate_function=surrogate.ATan(),
        #                                           detach_reset=True)
        self.fc2 = nn.Linear(256, num_classes)

        # for m in self.modules():
        #     if isinstance(m, Bottleneck):
        #         nn.init.constant_(m.bn3.weight, 0)
        #     elif isinstance(m, BasicBlock):
        #         nn.init.constant_(m.bn2.weight, 0)
        #     elif isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, planes, num_blocks, n_w, n_u, n_b, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, n_w, n_u, n_b, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def reset_dynamics(self):
        for m in self.modules():
            if isinstance(m,QConv2dLIF):
                m.lif_module.reset_mem()
            elif isinstance(m,QConvBN2dLIF):
                m.lif_module.reset_mem()
            elif isinstance(m,LIFSpike):
                m.reset_mem()
        self.direct_lif.reset_mem()
        self.lif_fc.reset_mem()
        return 0

    def weight_init(self):
        for m in self.modules():
            if isinstance(m,QConvBN2dLIF):
                nn.init.kaiming_uniform_(m.conv_module.weight)
                nn.init.kaiming_uniform_(m.bn_module.weight)
            elif isinstance(m,QConvBN2d):
                nn.init.kaiming_uniform_(m.conv_module.weight)
                nn.init.kaiming_uniform_(m.bn_module.weight)
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):

        # acc_voltage = 0
        u_out = []
        self.reset_dynamics()
        if args.dataset != 'dvs':
            static_x = self.bn1(self.conv1(x))
        # static_x = self.bn1(self.conv1(x))

        for t in range(self.total_timestep):
            if args.dataset == 'dvs':
                out = x[:,t].to(torch.float32).cuda()
                out = self.ConvBnLif1(out)
            else:
                out = self.direct_lif.direct_forward(static_x,False,0)
            # out = self.direct_lif.direct_forward(static_x,False,0)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.lif_fc(self.fc1(out),False,0,bias=0)
            out = self.fc2(out)

            # acc_voltage = acc_voltage + out
            u_out += [out]

        # acc_voltage = acc_voltage / self.total_timestep

        return u_out


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet19(num_classes, total_timestep):
    return ResNet(BasicBlock, [3,3,2], num_classes, total_timestep)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

