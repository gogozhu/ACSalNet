# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import *
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.utils import setup_seed
from utils.utils import WarmupMultiStepLR

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--m', dest='mention',
                        help='experiment mention',
                        default='', type=str)


    # philly
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


def main():
    args = parse_args()
    update_config(cfg, args)

    setup_seed(cfg.SEED)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in cfg.GPUS])

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, args.mention, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )
    # print(model)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
    )

    try:
        writer_dict['writer'].add_graph(model, (dump_input, ))
    except Exception as e:
        logger.info(e)

    try:
        logger.info(get_model_summary(model, dump_input))
    except:
        pass

    model = torch.nn.DataParallel(model, device_ids=list(range(len(cfg.GPUS)))).cuda()

    # define loss function (criterion) and optimizer
    criterion = eval(cfg.LOSS.NAME)(cfg).cuda()

    if cfg.LOSS.NAME == 'ModMSE_KL_CC_NSS_Loss':
        criterion_val = ModMSE_KL_CC_Loss(cfg).cuda()
    else:
        criterion_val = criterion

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    logger.info(os.linesep + 'train_set : {:d} entries'.format(len(train_dataset)))
    logger.info('val_set   : {:d} entries'.format(len(valid_dataset)) + os.linesep)

    if cfg.DATASET.SAMPLER == "":
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
    elif cfg.DATASET.SAMPLER == "RandomIdentitySampler":
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=dataset.RandomIdentitySampler(train_dataset.images, cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS), cfg.DATASET.NUM_INSTANCES),
            batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS) // cfg.DATASET.NUM_INSTANCES,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            sampler=dataset.RandomIdentitySampler(valid_dataset.images, cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS), cfg.DATASET.NUM_INSTANCES),
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS) // cfg.DATASET.NUM_INSTANCES,
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY
        )
    else:
        assert False

    best_perf = None
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    if cfg.TRAIN.WARMUP_EPOCHS > 0:
        lr_scheduler = WarmupMultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            warmup_iters=cfg.TRAIN.WARMUP_EPOCHS,
            last_epoch=last_epoch
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # torch.cuda.empty_cache()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # torch.cuda.empty_cache()

        # evaluate on validation set
        perf_indicator, is_larger_better = validate(
            cfg, valid_loader, valid_dataset, model, criterion_val,
            final_output_dir, tb_log_dir, writer_dict
        )

        if is_larger_better:
            if best_perf is None or perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
        else:
            if best_perf is None or perf_indicator <= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
