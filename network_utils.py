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
import math
from training_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
cudnn.deterministic = True

args = args_config.get_args()



class QFConvBN2dLIF(nn.Module):
    """ folding the conv2d and batchnorm2d in the inference"""

    def __init__(self, conv_module, bn_module, lif_module, num_bits_w=4, num_bits_bias=4, num_bits_u=4):
        super(QFConvBN2dLIF,self).__init__()

        self.conv_module = conv_module
        self.bn_module = bn_module
        self.lif_module = lif_module

        self.num_bits_w = num_bits_w
        self.num_bits_bias = num_bits_bias
        self.num_bits_u = num_bits_u
        
        # initial_w = conv_module.weight.data.abs().max()
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        # print(initial_w)
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()
        # nn.ParameterList([nn.Parameter(initial_w) for i in range(1)]).cuda()
        # print(self.scaling[0])
        # self.scaling = nn.Parameter(,requires_grad=True).cuda()
        # print(self.scaling)
        # self.scaling.requires_grad_(requires_grad=True)
    
    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std   
            weight = self.conv_module.weight * gamma_.reshape(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            # print(std.shape)
            # print(self.conv_module.weight.shape)
            weight = self.conv_module.weight * gamma_.reshape(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
                # bias = 0*mean
            
        return weight, bias


    def forward(self, x):
        if self.training:  
            ### Get the bn stats first, doing conv first
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias, 
                            stride=self.conv_module.stride,
                            padding=self.conv_module.padding,
                            dilation=self.conv_module.dilation,
                            groups=self.conv_module.groups)

            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.reshape(self.conv_module.out_channels, -1) # CNHW -> (C,NHW)
            mean = y.mean(1)
            var = y.var(1)

            self.bn_module.running_mean = \
                self.bn_module.momentum * self.bn_module.running_mean + \
                (1 - self.bn_module.momentum) * mean
            self.bn_module.running_var = \
                self.bn_module.momentum * self.bn_module.running_var + \
                (1 - self.bn_module.momentum) * var

        else:
            #### Using long term mean and var during inference
            mean = self.bn_module.running_mean
            var = self.bn_module.running_var

        std = torch.sqrt(var + self.bn_module.eps)
        weight, bias = self.fold_bn(mean, std)

        # print("w max:", weight.max())
        # print("b max:", bias.max())
        # print("w min:", weight.min())
        # print("b min:", bias.min())
        # if self.scaling is None:
        #     
        #     self.scaling = nn.ParameterList([nn.Parameter(torch.tensor([alpha])) for i in range(1)]).cuda()
        # else:
        #     qweight = w_q(weight, self.num_bits_w, self.scaling[0])
        if args.wq:
            if args.share:
                qweight,beta = w_q(weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(weight, self.num_bits_w)
        else:
            qweight = weight

        if args.bq:
            if args.share:
                qbias,beta = w_q(bias, self.num_bits_bias, beta)
            else:
                qbias = b_q(bias, self.num_bits_bias)
        else:
            qbias = bias

        x = F.conv2d(x, qweight,qbias,
                        stride=self.conv_module.stride,
                        padding=self.conv_module.padding,
                        dilation=self.conv_module.dilation,
                        groups=self.conv_module.groups)

        if args.share:
            s = self.lif_module(x, args.share, beta)
        else:
            s = self.lif_module(x, args.share, 0)

        return s


class QConv2dLIF(nn.Module):
    """ integerate the conv2d and LIF in the inference"""

    def __init__(self, conv_module, lif_module, num_bits_w=4, num_bits_u=4):
        super(QConv2dLIF,self).__init__()

        self.conv_module = conv_module
        self.lif_module = lif_module

        self.num_bits_w = num_bits_w
        self.num_bits_u = num_bits_u
        
        # initial_w = conv_module.weight.data.abs().max()
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        # print(initial_w)
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()
        # nn.ParameterList([nn.Parameter(initial_w) for i in range(1)]).cuda()
        # print(self.scaling[0])
        # self.scaling = nn.Parameter(,requires_grad=True).cuda()
        # print(self.scaling)
        # self.scaling.requires_grad_(requires_grad=True)
    



    def forward(self, x):
        # if self.training:
        if args.wq:
            if args.share:
                qweight,beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight
        # qweight= w_q(self.weight, self.num_bits_weight, in_alpha)
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
                                        padding=self.conv_module.padding,
                                        dilation=self.conv_module.dilation,
                                        groups=self.conv_module.groups)

        if args.share:
            s = self.lif_module(x, args.share, beta, bias=0)
        else:
            s = self.lif_module(x, args.share, 0, bias=0)
        # else:
        #     if args.wq:
        #         if args.share:
        #             qweight,beta = w_q_inference(self.conv_module.weight, self.num_bits_w, self.beta[0])
        #         else:
        #             qweight = b_q_inference(self.conv_module.weight, self.num_bits_w)
        #     else:
        #         qweight = self.conv_module.weight
        #     # print(torch.unique(qweight).shape)
        #     # qweight= w_q(self.weight, self.num_bits_weight, in_alpha)
        #     x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
        #                                     padding=self.conv_module.padding,
        #                                     dilation=self.conv_module.dilation,
        #                                     groups=self.conv_module.groups)

        #     if args.share:
        #         s = self.lif_module(x, args.share, beta, bias=0)
        #     else:
        #         s = self.lif_module(x, args.share, 0, bias=0)

        return s


class QConvBN2dLIF(nn.Module):
    """ integerate the conv2d, BN, and LIF in the inference"""

    def __init__(self, conv_module, bn_module, lif_module, num_bits_w=4,num_bits_b=4, num_bits_u=4):
        super(QConvBN2dLIF,self).__init__()

        self.conv_module = conv_module
        self.lif_module = lif_module
        self.bn_module  = bn_module

        self.num_bits_w = num_bits_w
        self.num_bits_b = num_bits_b
        self.num_bits_u = num_bits_u
        
        # initial_w = conv_module.weight.data.abs().max()
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        # print(initial_w)
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()
        # nn.ParameterList([nn.Parameter(initial_w) for i in range(1)]).cuda()
        # print(self.scaling[0])
        # self.scaling = nn.Parameter(,requires_grad=True).cuda()
        # print(self.scaling)
        # self.scaling.requires_grad_(requires_grad=True)
    



    def forward(self, x):
        # if self.training:
        if args.wq:
            if args.share:
                qweight,beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight
        # qweight= w_q(self.weight, self.num_bits_weight, in_alpha)
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
                                        padding=self.conv_module.padding,
                                        dilation=self.conv_module.dilation,
                                        groups=self.conv_module.groups)
        x = self.bn_module(x)
        # mean = self.bn_module.running_mean
        # var = self.bn_module.running_var
        # std = torch.sqrt(var + self.bn_module.eps)
        # gamma_ = (self.bn_module.weight / std)
        # bias = (self.bn_module.bias - gamma_ * mean)
        # gamma_ = gamma_.reshape(1,self.conv_module.out_channels, 1, 1)
        # bias = bias.reshape(1,self.conv_module.out_channels, 1, 1)
        # x = gamma_*x + bias
        if args.share:
            s = self.lif_module(x, args.share, beta, bias=0)
        else:
            s = self.lif_module(x, args.share, 0, bias=0)
        # else:
        #     if args.wq:
        #         if args.share:
        #             qweight,beta = w_q_inference(self.conv_module.weight, self.num_bits_w, self.beta[0])
        #         else:
        #             qweight,_ = b_q_inference(self.conv_module.weight, self.num_bits_w)
        #     else:
        #         qweight = self.conv_module.weight
        #     # print(torch.unique(qweight).shape)
        #     # qweight= w_q(self.weight, self.num_bits_weight, in_alpha)
        #     x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
        #                                     padding=self.conv_module.padding,
        #                                     dilation=self.conv_module.dilation,
        #                                     groups=self.conv_module.groups)
            
        #     # print(x.shape)
            
        #     mean = self.bn_module.running_mean
        #     var = self.bn_module.running_var
        #     std = torch.sqrt(var + self.bn_module.eps)
        #     gamma_ = (self.bn_module.weight / std)
        #     bias = (self.bn_module.bias - gamma_ * mean)
        #     gamma_ = gamma_.reshape(1,self.conv_module.out_channels, 1, 1)
        #     bias = bias.reshape(1,self.conv_module.out_channels, 1, 1)
            
        #     # print(gamma_.shape)
        #     if args.share:
        #         x = gamma_*x
        #     else:
        #         x = gamma_*x+ bias

        #     if args.share:
        #         s = self.lif_module(x, args.share, beta, bias=bias/beta)
        #     else:
        #         s = self.lif_module(x, args.share, 0)

        return s


class QConvBN2d(nn.Module):
    """ integerate the conv2d and BN in the inference"""

    def __init__(self, conv_module, bn_module, num_bits_w=4,num_bits_u=4,short_cut=False):
        super(QConvBN2d,self).__init__()

        self.conv_module = conv_module
        self.bn_module  = bn_module

        self.num_bits_w = num_bits_w
        self.num_bits_u = num_bits_u
        self.short_cut = short_cut

        
        initial_beta = torch.Tensor(conv_module.weight.abs().mean() * 2 / math.sqrt((2**(self.num_bits_w-1)-1)))
        self.beta = nn.ParameterList([nn.Parameter(initial_beta) for i in range(1)]).cuda()

    def forward(self, x):
        if args.wq:
            if args.share:
                qweight,beta = w_q(self.conv_module.weight, self.num_bits_w, self.beta[0])
            else:
                qweight = b_q(self.conv_module.weight, self.num_bits_w)
        else:
            qweight = self.conv_module.weight
        # qweight= w_q(self.weight, self.num_bits_weight, in_alpha)
        x = F.conv2d(x, qweight, self.conv_module.bias, stride=self.conv_module.stride,
                                        padding=self.conv_module.padding,
                                        dilation=self.conv_module.dilation,
                                        groups=self.conv_module.groups)
        x = self.bn_module(x)

        return x