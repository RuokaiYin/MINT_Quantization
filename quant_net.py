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



class Q_ShareScale_VGG9(nn.Module):
    def __init__(self,time_step,dataset):
        super(Q_ShareScale_VGG9, self).__init__()

        #### Set bitwidth for quantization
        self.num_bits_w = 2
        self.num_bits_u = 2
  
        #### Print out the parameters for quantization
        
        print("quant bw for w: " + str(self.num_bits_w))
        print("quant bw for u: " + str(self.num_bits_u))

        #### Other parameters for SNNs
        self.time_step = time_step

        input_dim = 3

        # print(args.th)


        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        self.direct_lif = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=False)
        # self.ConvBnLif1 = QConvBN2dLIF(conv1,bn1,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv2 = nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False)
        lif2 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvLif2 = QConv2dLIF(conv2,lif2,self.num_bits_w,self.num_bits_u)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        conv3 = nn.Conv2d(64,128, kernel_size=3, padding=1, bias=False)
        lif3 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvLif3 = QConv2dLIF(conv3,lif3,self.num_bits_w,self.num_bits_u)

        conv4 = nn.Conv2d(128,128, kernel_size=3, padding=1, bias=False)
        lif4 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvLif4 = QConv2dLIF(conv4,lif4,self.num_bits_w,self.num_bits_u)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        conv5 = nn.Conv2d(128,256, kernel_size=3, padding=1, bias=False)
        lif5 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvLif5 = QConv2dLIF(conv5,lif5,self.num_bits_w,self.num_bits_u)

        conv6 = nn.Conv2d(256,256, kernel_size=3, padding=1, bias=False)
        lif6 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvLif6 = QConv2dLIF(conv6,lif6,self.num_bits_w,self.num_bits_u)

        conv7 = nn.Conv2d(256,256, kernel_size=3, padding=1, bias=False)
        lif7 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvLif7 = QConv2dLIF(conv7,lif7,self.num_bits_w,self.num_bits_u)

        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        if dataset == 'tiny':
            size = 1
            clas = 200
        else:
            size = 1
            clas = 10
        self.fc_out = nn.Linear(256*(size**2), clas, bias=True)

        self.weight_init()

    def reset_dynamics(self):
        for m in self.modules():
            if isinstance(m,QConv2dLIF):
                m.lif_module.reset_mem()
        self.direct_lif.reset_mem()
        return 0

    def weight_init(self):
        for m in self.modules():
            if isinstance(m,QConvBN2dLIF):
                nn.init.kaiming_uniform_(m.conv_module.weight)
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


    def forward(self, inp):
        
        u_out = []
        self.reset_dynamics()
        static_input = self.conv1(inp)

        for t in range(self.time_step):
            s = self.direct_lif.direct_forward(static_input,False,0)

            s = self.ConvLif2(s)
            # print(torch.sum(s))
            s = self.pool1(s)

            s = self.ConvLif3(s)
            s = self.ConvLif4(s)

            s = self.pool2(s)

            s = self.ConvLif5(s)
            s = self.ConvLif6(s)
            s = self.ConvLif7(s)

            s = self.pool3(s)

            s = s.view(s.shape[0],-1)
            s = self.fc_out(s)

            u_out += [s]
    
        return u_out




class Q_ShareScale_VGG16(nn.Module):
    def __init__(self,time_step,dataset):
        super(Q_ShareScale_VGG16, self).__init__()

        #### Set bitwidth for quantization
        self.num_bits_w = 4
        self.num_bits_b = 4
        self.num_bits_u = 4
  
        #### Print out the parameters for quantization
        
        print("quant bw for w: " + str(self.num_bits_w))
        print("quant bw for b: " + str(self.num_bits_b))
        print("quant bw for u: " + str(self.num_bits_u))

        #### Other parameters for SNNs
        self.time_step = time_step

        if dataset == 'dvs':
            input_dim = 2
        else:
            input_dim = 3

        # print(args.th)

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine=True)
        self.direct_lif = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=False)
        # self.ConvBnLif1 = QConvBN2dLIF(conv1,bn1,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv1dvs = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        bn1dvs = nn.BatchNorm2d(64,affine=True)
        lif1dvs =  LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif1 = QConvBN2dLIF(conv1dvs,bn1dvs,lif1dvs,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv2 = nn.Conv2d(64,64, kernel_size=3, padding=1, bias=args.conv_b)
        bn2 = nn.BatchNorm2d(64,affine=args.bn_a)
        lif2 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif2 = QConvBN2dLIF(conv2,bn2,lif2,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        conv3 = nn.Conv2d(64,128, kernel_size=3, padding=1, bias=args.conv_b)
        bn3 = nn.BatchNorm2d(128,affine=args.bn_a)
        lif3 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif3 = QConvBN2dLIF(conv3,bn3,lif3,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv4 = nn.Conv2d(128,128, kernel_size=3, padding=1, bias=args.conv_b)
        bn4 = nn.BatchNorm2d(128,affine=args.bn_a)
        lif4 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif4 = QConvBN2dLIF(conv4,bn4,lif4,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        conv5 = nn.Conv2d(128,256, kernel_size=3, padding=1, bias=args.conv_b)
        bn5 = nn.BatchNorm2d(256,affine=args.bn_a)
        lif5 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif5 = QConvBN2dLIF(conv5,bn5,lif5,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv6 = nn.Conv2d(256,256, kernel_size=3, padding=1, bias=args.conv_b)
        bn6 = nn.BatchNorm2d(256,affine=args.bn_a)
        lif6 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif6 = QConvBN2dLIF(conv6,bn6,lif6,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv7 = nn.Conv2d(256,256, kernel_size=3, padding=1, bias=args.conv_b)
        bn7 = nn.BatchNorm2d(256,affine=args.bn_a)
        lif7 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif7 = QConvBN2dLIF(conv7,bn7,lif7,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        conv8 = nn.Conv2d(256,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn8 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif8 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif8 = QConvBN2dLIF(conv8,bn8,lif8,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv9 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn9 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif9 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif9 = QConvBN2dLIF(conv9,bn9,lif9,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv10 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn10 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif10 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif10 = QConvBN2dLIF(conv10,bn10,lif10,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        conv11 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn11 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif11 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif11 = QConvBN2dLIF(conv11,bn11,lif11,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv12 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn12 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif12 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif12 = QConvBN2dLIF(conv12,bn12,lif12,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv13 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn13 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif13 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif13 = QConvBN2dLIF(conv13,bn13,lif13,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool5 = nn.AvgPool2d(kernel_size=2)

        if dataset == 'tiny':
            size = 2
            clas = 200
        else:
            size = 1
            clas = 10
        self.fc_out = nn.Linear(512*(size**2), clas, bias=True)

        self.dataset = dataset

        self.weight_init()

    def reset_dynamics(self):
        for m in self.modules():
            if isinstance(m,QConvBN2dLIF):
                m.lif_module.reset_mem()
        self.direct_lif.reset_mem()
        return 0

    def weight_init(self):
        for m in self.modules():
            if isinstance(m,QConvBN2dLIF):
                nn.init.kaiming_uniform_(m.conv_module.weight)
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


    def forward(self, inp):
        
        u_out = []
        self.reset_dynamics()
        if self.dataset != 'dvs':
            static_input = self.bn1(self.conv1(inp))

        for t in range(self.time_step):
            if self.dataset == 'dvs':
                s = inp[:,t].to(torch.float32).cuda()
                s = self.ConvBnLif1(s)
            else:
                s = self.direct_lif.direct_forward(static_input,False,0)

            s = self.ConvBnLif2(s)
            # print(torch.sum(s))
            s = self.pool1(s)

            s = self.ConvBnLif3(s)
            s = self.ConvBnLif4(s)

            s = self.pool2(s)

            s = self.ConvBnLif5(s)
            s = self.ConvBnLif6(s)
            s = self.ConvBnLif7(s)

            s = self.pool3(s)

            s = self.ConvBnLif8(s)
            s = self.ConvBnLif9(s)
            s = self.ConvBnLif10(s)

            s = self.pool4(s)

            s = self.ConvBnLif11(s)
            s = self.ConvBnLif12(s)
            s = self.ConvBnLif13(s)
            # print(torch.sum(s))
            s = self.pool5(s)
            s = s.view(s.shape[0],-1)
            s = self.fc_out(s)

            u_out += [s]
    
        return u_out





class Q_ShareScale_Fold_VGG16(nn.Module):
    def __init__(self,time_step,dataset):
        super(Q_ShareScale_Fold_VGG16, self).__init__()

        #### Set bitwidth for quantization
        self.num_bits_w = 8
        self.num_bits_b = 8
        self.num_bits_u = 8
  
        #### Print out the parameters for quantization
        
        print("quant bw for w: " + str(self.num_bits_w))
        print("quant bw for b: " + str(self.num_bits_b))
        print("quant bw for u: " + str(self.num_bits_u))

        #### Other parameters for SNNs
        self.time_step = time_step

        input_dim = 3

        # print(args.th)

        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine=True)
        self.direct_lif = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=False)
        # self.ConvBnLif1 = QConvBN2dLIF(conv1,bn1,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv2 = nn.Conv2d(64,64, kernel_size=3, padding=1, bias=args.conv_b)
        bn2 = nn.BatchNorm2d(64,affine=args.bn_a)
        lif2 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif2 = QConvBN2dLIF(conv2,bn2,lif2,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        conv3 = nn.Conv2d(64,128, kernel_size=3, padding=1, bias=args.conv_b)
        bn3 = nn.BatchNorm2d(128,affine=args.bn_a)
        lif3 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif3 = QConvBN2dLIF(conv3,bn3,lif3,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv4 = nn.Conv2d(128,128, kernel_size=3, padding=1, bias=args.conv_b)
        bn4 = nn.BatchNorm2d(128,affine=args.bn_a)
        lif4 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif4 = QConvBN2dLIF(conv4,bn4,lif4,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        conv5 = nn.Conv2d(128,256, kernel_size=3, padding=1, bias=args.conv_b)
        bn5 = nn.BatchNorm2d(256,affine=args.bn_a)
        lif5 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif5 = QConvBN2dLIF(conv5,bn5,lif5,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv6 = nn.Conv2d(256,256, kernel_size=3, padding=1, bias=args.conv_b)
        bn6 = nn.BatchNorm2d(256,affine=args.bn_a)
        lif6 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif6 = QConvBN2dLIF(conv6,bn6,lif6,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv7 = nn.Conv2d(256,256, kernel_size=3, padding=1, bias=args.conv_b)
        bn7 = nn.BatchNorm2d(256,affine=args.bn_a)
        lif7 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif7 = QConvBN2dLIF(conv7,bn7,lif7,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        conv8 = nn.Conv2d(256,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn8 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif8 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif8 = QConvBN2dLIF(conv8,bn8,lif8,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv9 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn9 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif9 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif9 = QConvBN2dLIF(conv9,bn9,lif9,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv10 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn10 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif10 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif10 = QConvBN2dLIF(conv10,bn10,lif10,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        conv11 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn11 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif11 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif11 = QConvBN2dLIF(conv11,bn11,lif11,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv12 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn12 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif12 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif12 = QConvBN2dLIF(conv12,bn12,lif12,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        conv13 = nn.Conv2d(512,512, kernel_size=3, padding=1, bias=args.conv_b)
        bn13 = nn.BatchNorm2d(512,affine=args.bn_a)
        lif13 = LIFSpike(thresh=args.th, leak=args.leak_mem, gamma=1.0, soft_reset=args.sft_rst, quant_u=args.uq, num_bits_u=self.num_bits_u)
        self.ConvBnLif13 = QConvBN2dLIF(conv13,bn13,lif13,self.num_bits_w,self.num_bits_b,self.num_bits_u)

        self.pool5 = nn.AvgPool2d(kernel_size=2)

        if dataset == 'tiny':
            size = 2
            clas = 200
        else:
            size = 1
            clas = 10
        self.fc_out = nn.Linear(512*(size**2), clas, bias=True)

        self.weight_init()

    def reset_dynamics(self):
        for m in self.modules():
            if isinstance(m,QConvBN2dLIF):
                m.lif_module.reset_mem()
        self.direct_lif.reset_mem()
        return 0

    def weight_init(self):
        for m in self.modules():
            if isinstance(m,QConvBN2dLIF):
                nn.init.kaiming_uniform_(m.conv_module.weight)
            if isinstance(m,nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


    def forward(self, inp):
        
        u_out = []
        self.reset_dynamics()
        static_input = self.bn1(self.conv1(inp))

        for t in range(self.time_step):
            s = self.direct_lif(static_input,False,0)

            s = self.ConvBnLif2(s)
            # print(torch.sum(s))
            s = self.pool1(s)

            s = self.ConvBnLif3(s)
            s = self.ConvBnLif4(s)

            s = self.pool2(s)

            s = self.ConvBnLif5(s)
            s = self.ConvBnLif6(s)
            s = self.ConvBnLif7(s)

            s = self.pool3(s)

            s = self.ConvBnLif8(s)
            s = self.ConvBnLif9(s)
            s = self.ConvBnLif10(s)

            s = self.pool4(s)

            s = self.ConvBnLif11(s)
            s = self.ConvBnLif12(s)
            s = self.ConvBnLif13(s)
            # print(torch.sum(s))
            s = self.pool5(s)
            s = s.view(s.shape[0],-1)
            s = self.fc_out(s)

            u_out += [s]
    
        return u_out