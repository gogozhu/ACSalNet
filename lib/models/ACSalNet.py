# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhu Huansheng @ SCUT
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import math
from .dresnet import dresnet50
from torch.nn import functional as F
from .multi_heads_NL_attention import Multi_Heads_NL_Attention
from .dcn import DeformConv, ModulatedDeformConv


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}

class Pyramid_Context_Attention(nn.Module):
    """docstring for Pyramid_Context_Attention"""
    def __init__(self, size, pooltypes='avg'):
        super(Pyramid_Context_Attention, self).__init__()
        self.pooltypes = pooltypes
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.pool16 = nn.AvgPool2d(kernel_size=16, stride=16, padding=0)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out = x.mean(dim=1).unsqueeze(1)
        pf = [out]
        pf.append(F.upsample(self.pool4(out), size=(h, w), mode='bilinear', align_corners=True))
        pf.append(F.upsample(self.pool16(out), size=(h, w), mode='bilinear', align_corners=True))
        pf.append(F.upsample(self.GAP(out), size=(h, w), mode='bilinear', align_corners=True))
        out = self.fuse(torch.cat(pf, 1))
        return self.sigmoid(out) * x + x

class MSFE(nn.Module):
    def __init__(self, in_channel, planes, out_channel, rate):
        super(MSFE, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, planes, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        self.conv3 = nn.Conv2d(in_channel, planes, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        self.conv4 = nn.Conv2d(in_channel, planes, kernel_size=3, stride=1, padding=7, dilation=7, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(int(planes*4), out_channel*(rate**2), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel*(rate**2), momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(rate),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.fc1   = nn.Conv2d(planes*4, planes*4 // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(planes*4 // 16, out_channel, 1, bias=False)
        # self.fc2   = nn.Conv2d(planes*4 // 16, planes*4, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        fea_cat = torch.cat([x1, x2, x3, x4], 1)
        out = self.fuse(fea_cat)
        att = self.fc2(self.relu1(self.fc1(self.avg_pool(fea_cat))))
        return self.sigmoid(att) * out + out

    # def forward(self, x):
    #     x1 = self.conv1(x)
    #     x2 = self.conv2(x)
    #     x3 = self.conv3(x)
    #     x4 = self.conv4(x)
    #     fea_cat = torch.cat([x1, x2, x3, x4], 1)
    #     att = self.fc2(self.relu1(self.fc1(self.avg_pool(fea_cat))))
    #     fea_cat = self.sigmoid(att) * fea_cat + fea_cat
    #     out = self.fuse(fea_cat)
    #     return out

class HighResolutionModule(nn.Module):
    def __init__(self, cfg, block, num_blocks, scales, num_inchannels,
                num_channels, fuse_method):
        super(HighResolutionModule, self).__init__()

        self.cfg = cfg
        self.block = block
        self.num_blocks = num_blocks
        self.num_inchannels = num_inchannels
        self.num_channels = num_channels
        self.fuse_method = fuse_method

        assert len(num_inchannels)==2 and num_inchannels[0] <= num_inchannels[1]

        self.stage = None
        num_channels = num_channels // block.expansion
        if fuse_method == 'SUM':
            self.stage = self._make_stage(block, num_blocks, num_inchannels[0], num_channels)
        elif fuse_method == 'CONCAT':
            self.stage = self._make_stage(block, num_blocks, num_inchannels[0]*2, num_channels)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, block, num_blocks, num_inchannel, num_channels):
        downsample = None
        stride = 1
        if num_inchannel != num_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    num_inchannel,
                    num_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []

        layers.append(
            block(
                num_inchannel,
                num_channels,
                stride,
                downsample
            ) if block == BasicBlock or block == Bottleneck else block(
                num_inchannel,
                num_channels,
                16, 4,
                stride,
                downsample,
                base_width=8
            )
        )
        nl_attention=dict(num_heads=8, kv_stride=4)
        layers.append(Multi_Heads_NL_Attention(num_channels, **nl_attention))

        num_inchannel = num_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    num_inchannel,
                    num_channels
                ) if block == BasicBlock or block == Bottleneck else block(
                    num_inchannel,
                    num_channels,
                    16, 4,
                    base_width=8
                )
            )
            layers.append(Pyramid_Context_Attention(size=self.cfg.MODEL.IMAGE_SIZE))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_fuse = None
        if self.fuse_method == 'SUM':
            x_fuse = x[0] + x[1]
        elif self.fuse_method == 'CONCAT':
            x_fuse = torch.cat(x, 1)
        x_fuse = self.stage(x_fuse)
        return x_fuse

class Upsampler(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsampler, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*4, kernel_size=3, stride=1,
                               padding=1, bias=False),
            nn.BatchNorm2d(out_channel * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        out = self.up(x)
        return out

class ACSalNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        self.cfg = cfg
        extra = cfg.MODEL.EXTRA
        super(ACSalNet, self).__init__()

        resnet = dresnet50(pretrained=True)
        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = \
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool), \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        fpn_channel_num = extra.FPN_CHANNEL_NUM

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, fpn_channel_num, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, fpn_channel_num, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, fpn_channel_num, kernel_size=1, stride=1, padding=0)

        msfe_out_channel_num = extra.MSFE_OUT_CHANNEL_NUM
        msfe_inner_channels = extra.MSFE_INNER_CHANNEL_NUM

        nl_attention=dict(num_heads=8, kv_stride=2)
        self.smooth0 = nn.Sequential(
            MSFE(fpn_channel_num, msfe_inner_channels[0], msfe_out_channel_num, rate=8),
            Multi_Heads_NL_Attention(msfe_out_channel_num, **nl_attention)
        )
        self.smooth1 =  nn.Sequential(
            MSFE(fpn_channel_num, msfe_inner_channels[1], msfe_out_channel_num, rate=4),
            Multi_Heads_NL_Attention(msfe_out_channel_num, **nl_attention)
        )
        self.smooth2 =  nn.Sequential(
            MSFE(fpn_channel_num, msfe_inner_channels[2], msfe_out_channel_num, rate=2),
            Multi_Heads_NL_Attention(msfe_out_channel_num, **nl_attention)
        )
        self.smooth3 =  nn.Sequential(
            MSFE(fpn_channel_num, msfe_inner_channels[3], msfe_out_channel_num, rate=1),
            Multi_Heads_NL_Attention(msfe_out_channel_num, **nl_attention)
        )

        stage2_cfg = extra.STAGE2
        stage3_cfg = extra.STAGE3
        stage4_cfg = extra.STAGE4

        self.stage2 = HighResolutionModule(
            self.cfg,
            blocks_dict[stage2_cfg['BLOCK']],
            stage2_cfg['NUM_BLOCKS'],
            stage2_cfg['SCALES'],
            stage2_cfg['NUM_CHANNELS'],
            stage3_cfg['NUM_CHANNELS'][0],
            stage2_cfg['FUSE_METHOD']
        )

        self.stage3 = HighResolutionModule(
            self.cfg,
            blocks_dict[stage3_cfg['BLOCK']],
            stage3_cfg['NUM_BLOCKS'],
            stage3_cfg['SCALES'],
            stage3_cfg['NUM_CHANNELS'],
            stage4_cfg['NUM_CHANNELS'][0],
            stage3_cfg['FUSE_METHOD']
        )

        self.stage4 = HighResolutionModule(
            self.cfg,
            blocks_dict[stage4_cfg['BLOCK']],
            stage4_cfg['NUM_BLOCKS'],
            stage4_cfg['SCALES'],
            stage4_cfg['NUM_CHANNELS'],
            stage4_cfg['NUM_CHANNELS'][0],
            stage4_cfg['FUSE_METHOD']
        )

        last_channel = stage4_cfg['NUM_CHANNELS'][0] if self.cfg.MODEL.EXTRA.USE_HIGH_RES_PATH else msfe_out_channel_num * 4

        self.deconv_layers = None
        if extra.DECONV.NUM != 0:
            self.deconv_layers = self._make_deconv_layer( \
                extra.DECONV.NUM, \
                extra.DECONV.NUM_CHANNELS, \
                extra.DECONV.KERNELS,
                last_channel
            )
            last_channel = extra.DECONV.NUM_CHANNELS[-1]

        self.final_layer = []
        for _ in range(extra.DECONV.EXTRA_LAYERS):
            self.final_layer.append(
                BasicBlock(last_channel, last_channel)
            )
        self.final_layer.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=last_channel,
                    out_channels=1,
                    kernel_size=extra.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
                ),
                # nn.ReLU(inplace=True)
            )
        )
        self.final_layer = nn.Sequential(*self.final_layer)

        in_planes = fpn_channel_num
        self.ppms_pre = nn.Conv2d(2048, in_planes, 1, 1, bias=False)
        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(in_planes, in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)
        self.ppm_cat = nn.Sequential(nn.Conv2d(in_planes * 4, in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))

    def _upsample_sum(self, x, y, func=None):
        _,_,H,W = y.size()
        if func is None:
            return F.upsample(x, size=(H,W), mode='bilinear', align_corners=True) + y
        else:
            return func(F.upsample(x, size=(H,W), mode='bilinear', align_corners=True) + y)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, in_channel):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(nn.Sequential(
                Upsampler(in_channel, planes),
                nn.Conv2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                # nn.ReLU(inplace=True)
            ))
            in_channel = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        xs = self.ppms_pre(l4)
        xls = [xs]
        for k in range(len(self.ppms)):
            xls.append(F.upsample(self.ppms[k](xs), xs.size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))

        # Top-down
        p5 = xls
        p4 = self._upsample_sum(p5, self.latlayer1(l3))
        p3 = self._upsample_sum(p4, self.latlayer2(l2))
        p2 = self._upsample_sum(p3, self.latlayer3(l1))

        # Smooth
        l4 = self.smooth0(p5)
        l3 = self.smooth1(p4)
        l2 = self.smooth2(p3)
        l1 = self.smooth3(p2)

        if self.cfg.MODEL.EXTRA.USE_HIGH_RES_PATH:
            s2 = self.stage2([l1, l2])
            s3 = self.stage3([s2, l3])
            x = self.stage4([s3, l4])
        else:
            x = torch.cat([l4, l1, l2, l3], 1)

        if self.deconv_layers is not None:
            x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for need_init in [self.stage2, self.stage3, self.stage4, self.deconv_layers, self.final_layer, \
                self.ppms_pre, self.ppm_cat, self.ppms, self.smooth0, self.smooth1, self.smooth2, self.smooth3, \
                self.latlayer1, self.latlayer2, self.latlayer3]:
            if need_init is None:
                continue
            for m in need_init.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    # nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    for name, _ in m.named_parameters():
                        if name in ['bias']:
                            nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            # need_init_state_dict = {}
            # for name, m in pretrained_state_dict.items():
            #     if name.split('.')[0] in self.pretrained_layers \
            #        or self.pretrained_layers[0] is '*':
            #         need_init_state_dict[name] = m
            self.load_state_dict(pretrained_state_dict['best_state_dict'], strict=True)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

        if self.cfg.MODEL.FINETUNE:
            for param in self.parameters():
                param.requires_grad = False
            need_finetune = [self.deconv_layers, self.final_layer, \
                self.smooth0, self.smooth1, self.smooth2, self.ppms_pre, self.ppm_cat, self.ppms,
                self.latlayer1, self.latlayer2, self.layer4, self.layer3]
            for mm in need_finetune:
                for param in mm.parameters():
                    param.requires_grad = True

def get_pose_net(cfg, is_train, **kwargs):
    model = ACSalNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
