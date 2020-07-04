# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

EPSILON = np.finfo('float').eps

def KL_div(output, target, return_dense=False):
    output = output/torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(*output.shape)
    output = output/output.sum(2).sum(2).unsqueeze(2).unsqueeze(2).expand(*output.shape)
    if len(target.shape) < 4:
        target = target.unsqueeze(1)
    target = target/target.sum(2).sum(2).unsqueeze(2).unsqueeze(2).expand(*target.shape)
    a = target * torch.log(target/(output+1e-6)+1e-6)
    b = output * torch.log(output/(target+1e-6)+1e-6)
    if not return_dense:
        KL_Score = (a.sum()+b.sum())/(output.shape[0]*2.)
        # KL_Score = a.sum()
        if torch.isnan(KL_Score):
            KL_Score = 0
    else:
        KL_Score = (a.sum(-1).sum(-1) + b.sum(-1).sum(-1))/2.
        for i in range(KL_Score.size(0)):
            if torch.isnan(KL_Score[i]):
                KL_Score[i] = 0
    return KL_Score

def CC(output, target, return_dense=False):
    output = nn.functional.instance_norm(output)
    if len(target.shape) < 4:
        target = target.unsqueeze(1)
    target = nn.functional.instance_norm(target)
    num = (output * target).sum(3).sum(2)
    out_square = (output * output).sum(3).sum(2)
    tar_square = (target * target).sum(3).sum(2)
    if not return_dense:
        CC_score = (num/(torch.sqrt(out_square*tar_square)+1e-6)).mean()
        if torch.isnan(CC_score) or CC_score < 0:
            CC_score = 0
    else:
        CC_score = (num/(torch.sqrt(out_square*tar_square)+1e-6))
        for i in range(CC_score.size(0)):
            if torch.isnan(CC_score[i]) or CC_score[i] < 0:
                CC_score[i] = 0
    return CC_score

def NSS(output, target, fixationMap, return_dense=False):
    output = nn.functional.instance_norm(output)
    Sal = output*fixationMap*target
    fixationMap =fixationMap*target
    if not return_dense:
        NSS_score = (Sal.sum(-1).sum(-1)/fixationMap.sum(-1).sum(-1)).mean()
        if torch.isnan(NSS_score):
            NSS_score = 0
    else:
        NSS_score = (Sal.sum(-1).sum(-1)/fixationMap.sum(-1).sum(-1))
        for i in range(NSS_score.size(0)):
            if torch.isnan(NSS_score[i]):
                NSS_score[i] = 0
    return NSS_score

class ModMSELoss(nn.Module):
    def __init__(self, cfg):
        super(ModMSELoss, self).__init__()
        self.cfg = cfg

    def forward(self, outputs, label):
        if isinstance(outputs, list):
            output, prior = outputs
        else:
            output, prior = outputs, None
        loss = 0
        if prior is not None:
            prior_size = prior.shape
            reg = ( 1.0/(prior_size[-1]*prior_size[-2]) ) * prior**2
            loss += torch.sum(reg)
        output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
        # reg = ( 1.0/(prior_size[0]*prior_size[1]) ) * ( 1 - prior)**2
        loss += torch.mean( ((output / output_max) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )

        return loss, None

class ModMSE_KL_CC_Loss(nn.Module):
    def __init__(self, cfg):
        super(ModMSE_KL_CC_Loss, self).__init__()
        self.cfg = cfg
        self.return_dense = (cfg.TRAIN.OHEM != -1)

    def forward(self, outputs, label, fixmap):
        if isinstance(outputs, list):
            output, prior = outputs
            # assert False
        else:
            output, prior = outputs, None
        if prior is not None:
            prior_size = prior.shape
            reg = ( 1.0/(prior_size[-1]*prior_size[-2]) ) * prior**2
            loss_prior = torch.sum(reg)
        loss_KL = KL_div(output, label, self.return_dense)
        loss_CC = 1. - CC(output, label, self.return_dense)
        output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
        if self.return_dense:
            loss_MSE = (((output / output_max) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1)).mean(-1).mean(-1)
        else:
            loss_MSE = torch.mean( ((output / output_max) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )

        # loss_main = torch.mean( ((output / output.sum()) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
        # loss_main = torch.mean( (output - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
        if prior is not None:
            return loss_prior / 100 + \
                    loss_KL*self.cfg.LOSS.KL_ALPHA + \
                    loss_CC*self.cfg.LOSS.CC_ALPHA + \
                    loss_MSE*self.cfg.LOSS.MSE_ALPHA, \
                    '' if self.return_dense else '\t(P:{} KL:{:.3f} CC:{:.3f} MSE:{:.3f})'.format(loss_prior,loss_KL,1-loss_CC,loss_MSE)
        else:
            return loss_KL*self.cfg.LOSS.KL_ALPHA + \
                    loss_CC*self.cfg.LOSS.CC_ALPHA + \
                    loss_MSE*self.cfg.LOSS.MSE_ALPHA, \
                    '' if self.return_dense else '\t(KL:{:.3f} CC:{:.3f} MSE:{:.3f})'.format(loss_KL,1-loss_CC,loss_MSE)

class ModMSE_KL_CC_NSS_Loss(nn.Module):
    def __init__(self, cfg):
        super(ModMSE_KL_CC_NSS_Loss, self).__init__()
        self.cfg = cfg

    def forward(self, outputs, label, fixmap):
        if isinstance(outputs, list):
            output, prior = outputs
            # assert False
        else:
            output, prior = outputs, None
        if prior is not None:
            prior_size = prior.shape
            reg = ( 1.0/(prior_size[-1]*prior_size[-2]) ) * prior**2
            loss_prior = torch.sum(reg)
        loss_KL = KL_div(output, label)
        loss_CC = 1. - CC(output, label)
        loss_NSS = - NSS(output, label, fixmap)
        output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
        loss_MSE = torch.mean( ((output / output_max) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
        # loss_main = torch.mean( ((output / output.sum()) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
        # loss_main = torch.mean( (output - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
        if prior is not None:
            return loss_prior / 100 + \
                    loss_KL*self.cfg.LOSS.KL_ALPHA + \
                    loss_CC*self.cfg.LOSS.CC_ALPHA + \
                    loss_NSS*self.cfg.LOSS.NSS_ALPHA + \
                    loss_MSE*self.cfg.LOSS.MSE_ALPHA, \
                    '\t(P:{} KL:{:.3f} CC:{:.3f} NSS:{:.3f} MSE:{:.3f})'.format(loss_prior,loss_KL,1-loss_CC,-loss_NSS,loss_MSE)
        else:
            return loss_KL*self.cfg.LOSS.KL_ALPHA + \
                    loss_CC*self.cfg.LOSS.CC_ALPHA + \
                    loss_NSS*self.cfg.LOSS.NSS_ALPHA + \
                    loss_MSE*self.cfg.LOSS.MSE_ALPHA, \
                    '\t(KL:{:.3f} CC:{:.3f} NSS:{:.3f} MSE:{:.3f})'.format(loss_KL,1-loss_CC,-loss_NSS,loss_MSE)

class ModMSE_KL_CC_Loss_ML(nn.Module):
    def __init__(self, cfg):
        super(ModMSE_KL_CC_Loss_ML, self).__init__()
        self.cfg = cfg

    def forward(self, outputs, label, fixmap):
        if not isinstance(outputs, list):
            assert False
        losses = []
        final_loss = 0
        for idx, output in enumerate(outputs):
            _, _, h, w = output.shape
            if output.shape[-2:] != label.shape[-2:]:
                if len(label.shape) < 4:
                    label = label.unsqueeze(1)
                label = F.upsample_bilinear(label, [h,w])
            loss_KL = KL_div(output, label)
            loss_CC = 1. - CC(output, label)
            output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
            loss_MSE = torch.mean( ((output / output_max) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
            # loss_main = torch.mean( ((output / output.sum()) - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
            # loss_main = torch.mean( (output - label.unsqueeze(1))**2 / (1 - label.unsqueeze(1) + 0.1) )
            loss =  loss_KL*self.cfg.LOSS.KL_ALPHA + \
                    loss_CC*self.cfg.LOSS.CC_ALPHA + \
                    loss_MSE*self.cfg.LOSS.MSE_ALPHA
            final_loss += loss if len(self.cfg.LOSS.ALPHA_LOSS)==0 else loss*self.cfg.LOSS.ALPHA_LOSS[idx]
            losses.append(loss)
        return final_loss/len(losses), \
                '\t' + ' '.join(['L{}:{:.3f}'.format(i, l) for i, l in enumerate(losses)])


class MSE(nn.Module):
    """docstring for MSE"""
    def __init__(self, cfg):
        super(MSE, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, outputs, label, fixmap):
        return self.loss(outputs, label), ''

