import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_Heads_NL_Attention(nn.Module):

    def __init__(self, in_dim, num_heads=8, kv_stride=2):

        super(Multi_Heads_NL_Attention, self).__init__()

        self.num_heads = num_heads
        self.channel_in = in_dim
        self.kv_stride = kv_stride
        self.qk_embed_dim = in_dim // num_heads
        channel_out = self.qk_embed_dim * num_heads

        self.key_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=channel_out,
            kernel_size=1,
            bias=False)
        self.key_conv.kaiming_init = True

        self.v_dim = in_dim // num_heads
        self.value_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=self.v_dim * num_heads,
            kernel_size=1,
            bias=False)
        self.value_conv.kaiming_init = True

        stdv = 1.0 / math.sqrt(self.qk_embed_dim * 2)
        appr_bias_value = -2 * stdv * torch.rand(channel_out) + stdv
        self.appr_bias = nn.Parameter(appr_bias_value)

        self.proj_conv = nn.Conv2d(
            in_channels=self.v_dim * num_heads,
            out_channels=in_dim,
            kernel_size=1,
            bias=True)
        self.proj_conv.kaiming_init = True
        self.gamma = nn.Parameter(torch.zeros(1))

        self.q_downsample = None

        if self.kv_stride > 1:
            self.kv_downsample = nn.AvgPool2d(
                kernel_size=self.kv_stride, stride=self.kv_stride)
        else:
            self.kv_downsample = None

        self.init_weights()

    def forward(self, x_input):
        num_heads = self.num_heads

        x_q = x_input
        n, _, h, w = x_q.shape

        if self.kv_downsample is not None:
            x_kv = self.kv_downsample(x_input)
        else:
            x_kv = x_input
        _, _, h_kv, w_kv = x_kv.shape

        proj_key = self.key_conv(x_kv).view(
            (n, num_heads, self.qk_embed_dim, h_kv * w_kv))

        appr_bias = self.appr_bias.\
            view(1, num_heads, 1, self.qk_embed_dim).\
            repeat(n, 1, 1, 1)

        energy = torch.matmul(appr_bias, proj_key).\
            view(n, num_heads, 1, h_kv * w_kv)
        
        attention = F.softmax(energy, 3)

        proj_value = self.value_conv(x_kv)
        proj_value_reshape = proj_value.\
            view((n, num_heads, self.v_dim, h_kv * w_kv)).\
            permute(0, 1, 3, 2)

        out = torch.matmul(attention, proj_value_reshape).\
            permute(0, 1, 3, 2).\
            contiguous().\
            view(n, self.v_dim * self.num_heads, 1, 1)

        out = self.proj_conv(out)
        out = self.gamma * out + x_input
        return out

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'kaiming_init') and m.kaiming_init:
                nn.init.kaiming_uniform_(
                    m.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
