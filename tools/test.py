# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import *
from core.function import validate
from core.function import test
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--s', dest='session',
                        help='session to load model',
                        default=0, type=int)

    parser.add_argument('--mt', dest='modelType',
                        help='best model or final model',
                        type=str,
                        choices=['best','final','current'],
                        default='')

    parser.add_argument('--mode', dest='mode',
                        help='valid or test',
                        type=str,
                        choices=['val','test'],
                        default='val')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args

modelDict = {"best": "model_best.pth",
            "final": "final_state.pth",
            "current": "checkpoint.pth"}

def main():
    args = parse_args()
    update_config(cfg, args)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in cfg.GPUS])

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, phase='valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, modelDict[args.modelType]
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        if 'current' == args.modelType:
            model.load_state_dict(torch.load(model_state_file)['best_state_dict'], strict=True)
        else:
            model.load_state_dict(torch.load(model_state_file)['best_state_dict'], strict=True)

    model = torch.nn.DataParallel(model, device_ids=list(range(len(cfg.GPUS)))).cuda()

    # define loss function (criterion) and optimizer

    if cfg.LOSS.NAME == 'ModMSE_KL_CC_NSS_Loss':
        criterion = ModMSE_KL_CC_Loss(cfg).cuda()
    else:
        criterion = eval(cfg.LOSS.NAME)(cfg).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if args.mode == 'val':
        valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, 'val', False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        if cfg.DATASET.SAMPLER == "":
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=True
            )
        elif cfg.DATASET.SAMPLER == "RandomIdentitySampler":
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                sampler=dataset.RandomIdentitySampler(valid_dataset.images, cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS), cfg.DATASET.NUM_INSTANCES),
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=True
            )
        else:
            assert False
        # evaluate on validation set
        perf_indicator, res = validate(cfg, valid_loader, valid_dataset, model, criterion,
                 final_output_dir, tb_log_dir, returnRes=True)

        valid_dataset.evaluate(final_output_dir, res, modelDict[args.modelType].split('.')[0])
    else:
        test_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
            cfg, cfg.DATASET.ROOT, 'test', False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        if cfg.DATASET.SAMPLER == "":
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=True
            )
        elif cfg.DATASET.SAMPLER == "RandomIdentitySampler":
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                sampler=dataset.RandomIdentitySampler(test_dataset.images, cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS), cfg.DATASET.NUM_INSTANCES),
                batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS) // cfg.DATASET.NUM_INSTANCES,
                shuffle=False,
                num_workers=cfg.WORKERS,
                pin_memory=True
            )
        else:
            assert False
        output_dir = os.path.join(final_output_dir, cfg.TEST.OUT_DIR)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        test(cfg, test_loader, model, output_dir)


if __name__ == '__main__':
    main()
