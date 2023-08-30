import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training_utils import *
import tracemalloc
import gc

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = (input > 0).float()
        # L = torch.tensor([gamma])
        ctx.save_for_backward(input)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        grad_input = grad_input * ((1.0 - torch.abs(input)).clamp(min=0))

        # tmp = torch.ones_like(input)
        # tmp = torch.where(input.abs() < 0.5, 1., 0.)
        # grad_input = grad_input*torch.where(torch.abs(input-th)<1., 1, 0)
        # grad_input = grad_input*torch.where(input.abs() < 1., 1., 0.)
        return grad_input




class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, leak=0.5, gamma=1.0, soft_reset=True, quant_u=False, num_bits_u=4):
        """
        Implementing the LIF neurons.
        @param thresh: firing threshold;
        @param tau: membrane potential decay factor;
        @param gamma: hyper-parameter for controlling the sharpness in surrogate gradient;
        @param soft_reset: whether using soft-reset or hard-reset.
        """
        super(LIFSpike, self).__init__()

        # self.act = ZIF.apply
        self.quant_u = quant_u
        self.num_bits_u = num_bits_u

        self.thresh = thresh
        self.leak = leak
        self.gamma = gamma
        self.soft_reset = soft_reset

        self.membrane_potential = 0
        # print(self.thresh)

    def reset_mem(self):
        self.membrane_potential = 0

    def forward(self, s, share, beta, bias):
        # act = ZIF.apply
        # if self.training:
        # beta*
        # x = gamma_*x + bias
        H = s + self.membrane_potential
        
        # s = act(H-self.thresh)
        grad = ((1.0 - torch.abs(H-self.thresh)).clamp(min=0))
        s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad.detach()
        # s = (H -(self.thresh/beta)>0).float()
        if self.soft_reset:
            U = (H - s*self.thresh)*self.leak
        else:
            U = H*self.leak*(1-s)
        
        if self.quant_u:
            if share:
                self.membrane_potential = u_q(U,self.num_bits_u,beta)
            else:
                self.membrane_potential= b_q(U,self.num_bits_u)
        else:
            self.membrane_potential = U
        # else:
        #     # print(torch.unique(s).shape)
            
        #     H = s + self.membrane_potential
        #     # print(torch.unique(H).shape)
        #     if share:
        #         # s = torch.zeros_like(H).cuda()
        #         # s[H >(self.thresh/beta-bias)] = 1.0
        #         s = (H -(self.thresh/beta-bias)>0).float()
        #     else:
        #         s = ((H-self.thresh) > 0).float()
        #     # print(torch.unique(H).shape)
        #     if self.soft_reset:
        #         U = (H - s*self.thresh)*self.leak
        #     else:
        #         U = H*self.leak*(1-s)
            
        #     # if self.quant_u:
        #     #     if share:
        #     #         self.membrane_potential = (U).round().clamp(min=2**(-(self.num_bits_u-1)),max=2**(self.num_bits_u-1)-1)
        #     # # if self.quant_u:
        #     # #     if share:
        #     # #         self.membrane_potential,_ = b_q_inference(U,self.num_bits_u)
        #     # #     else:
        #     # #         self.membrane_potential = b_q_inference(U,self.num_bits_u)
        #     # else:
        #     self.membrane_potential = U
            
        return s
    
    def direct_forward(self, s, share, beta):
        # act = ZIF.apply
        # if self.training:
        # beta*
        # x = gamma_*x + bias
        H = s + self.membrane_potential
        
        # s = act(H-self.thresh)
        grad = ((1.0 - torch.abs(H-self.thresh)).clamp(min=0))
        s = (((H-self.thresh) > 0).float() - H*grad).detach() + H*grad.detach()
        if self.soft_reset:
            U = (H - s*self.thresh)*self.leak
        else:
            U = H*self.leak*(1-s)
        
        if self.quant_u:
            if share:
                self.membrane_potential = u_q(U,self.num_bits_u,beta)
            else:
                self.membrane_potential= b_q(U,self.num_bits_u)
        else:
            self.membrane_potential = U
        
        return s