# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import pprint
import numpy as np
import random
import adabound
from .AdamW import AdamW

from torch.optim.lr_scheduler import MultiStepLR


class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=0.1,
                 warmup_iters=5, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        lr = super().get_lr()
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [l * warmup_factor for l in lr]
        return lr

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def create_logger(cfg, cfg_name, mention='', phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)

    if phase == 'train':
        # set up logger
        if not root_output_dir.exists():
            print('=> creating {}'.format(root_output_dir))
            root_output_dir.mkdir()

        dataset = cfg.DATASET.DATASET
        dataset = dataset.replace(':', '_')
        model = cfg.MODEL.NAME
        cfg_name = os.path.basename(cfg_name).split('.')[0]

        final_output_dir = root_output_dir / dataset / model / cfg_name
        time_str = time.strftime('%Y-%m-%d-%H-%M')

        if not final_output_dir.exists():
            print('=> creating {}'.format(final_output_dir))
            final_output_dir.mkdir(parents=True, exist_ok=True)

        mention_file = final_output_dir / (str(cfg_name) + '_mentions.txt')
        session_info = {"time":time_str, "session":0, "cfg_name":cfg_name, "phase":phase, \
            "log_file":'{}_{}_{}.log'.format(cfg_name, time_str, phase),
            "tensorboard_log_dir":Path(cfg.LOG_DIR) / dataset / model,
            "mention":mention,
        }
        if not mention_file.exists():
            session_info["session"] = 1
            lines = []
        else:
            with mention_file.open('r') as f:
                line = f.readline()
                session_info["session"] = int(line.split(":")[-1])+1
                lines = f.readlines()

        with mention_file.open('w') as f:
            f.writelines("total session: {}\n".format(session_info["session"]))
            f.writelines(lines)
            f.writelines("\n\n" + "#"*80 + "\n\n")
            f.writelines(pprint.pformat(session_info))

        final_output_dir = final_output_dir / 's{}'.format(session_info["session"])
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

        log_file = session_info["log_file"]
        final_log_file = final_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        tensorboard_log_dir = session_info["tensorboard_log_dir"] / (cfg_name + '_s{}'.format(session_info["session"]) + '_' + time_str)

        print('=> creating {}'.format(tensorboard_log_dir))
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        return logger, str(final_output_dir), str(tensorboard_log_dir)
    else:
        assert root_output_dir.exists(), "{} should be a valid path".format(root_output_dir)
        dataset = cfg.DATASET.DATASET
        dataset = dataset.replace(':', '_')
        model = cfg.MODEL.NAME
        cfg_name = os.path.basename(cfg_name).split('.')[0]

        final_output_dir = root_output_dir / dataset / model / cfg_name / 's{}'.format(cfg.TEST.SESSION)
        time_str = time.strftime('%Y-%m-%d-%H-%M')

        assert final_output_dir.exists(), "{} should be a valid path".format(final_output_dir)

        log_file = '{}_{}.log'.format(phase, time_str)
        final_log_file = final_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(final_output_dir), ""


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.MODEL.FINETUNE:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam_amsgrad':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.TRAIN.LR,
                amsgrad=True
            )
        elif cfg.TRAIN.OPTIMIZER == 'adamw':
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.TRAIN.LR,
                weight_decay=1e-4,
            )
        elif cfg.TRAIN.OPTIMIZER == 'adabound':
            optimizer = adabound.AdaBound(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.TRAIN.LR,
                final_lr = 0.1,
            )
    else:
        if cfg.TRAIN.OPTIMIZER == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                momentum=cfg.TRAIN.MOMENTUM,
                weight_decay=cfg.TRAIN.WD,
                nesterov=cfg.TRAIN.NESTEROV
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.TRAIN.LR
            )
        elif cfg.TRAIN.OPTIMIZER == 'adam_amsgrad':
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                amsgrad=True
            )
        elif cfg.TRAIN.OPTIMIZER == 'adamw':
            optimizer = AdamW(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                weight_decay=1e-4,
            )
        elif cfg.TRAIN.OPTIMIZER == 'adabound':
            optimizer = adabound.AdaBound(
                model.parameters(),
                lr=cfg.TRAIN.LR,
                final_lr = 0.1,
            )

    return optimizer

def get_group_gn(cfg, dim):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    dim_per_gp = cfg.GROUP_NORM.DIM_PER_GP
    num_groups = cfg.GROUP_NORM.NUM_GROUPS

    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        # torch.save(states['best_state_dict'],
        torch.save(states,
                   os.path.join(output_dir, 'model_best.pth'))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, smap, fixmap, info in loader:
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
