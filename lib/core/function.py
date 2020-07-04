# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch
from torch.nn import functional as F
from utils.utils import clip_gradient


logger = logging.getLogger(__name__)

def ohem(loss, rate):
    top_k = int(loss.size(0) * rate)
    if top_k == 0:
        top_k = loss.size(0)
    topk_val, topk_idx = torch.topk(loss, k=top_k, dim=0, sorted=False)
    ohem_loss = torch.sum(torch.gather(loss, 0, topk_idx)) / top_k
    return ohem_loss


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    tit = 0
    for i, (input, smap, fixmap, info) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if tit == 0:
            optimizer.zero_grad()

        # compute output
        outputs = model(input)

        smap = smap.cuda(non_blocking=True)
        fixmap = fixmap.cuda(non_blocking=True)

        if config.DATASET.SAMPLER == "RandomIdentitySampler":
            outputs = outputs[0]

        loss, other_info = criterion(outputs, smap, fixmap)

        if config.TRAIN.OHEM != -1:
            loss = ohem(loss, config.TRAIN.OHEM)

        # measure accuracy and record loss
        losses.update(loss.item(), len(input) if isinstance(input, list) else input.size(0))

        loss /= config.TRAIN.ITER_NUM
        loss.backward()
        tit += 1

        if tit == config.TRAIN.ITER_NUM:
            # compute gradient and do update step
            if config.TRAIN.CLIP_GRAD != 0:
                clip_gradient(model, config.TRAIN.CLIP_GRAD)
            optimizer.step()
            tit = 0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            num_instances = len(input) if isinstance(input, list) else input.size(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        speed=num_instances/batch_time.val,
                        data_time=data_time, loss=losses)
            if other_info is not None:
                msg += other_info
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            # save_debug_images(config, input, info, target, pred*4, output,
            #                   prefix)

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None, returnRes=False):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)

    image_path = []
    filenames = []
    imgnums = []
    res = []
    with torch.no_grad():
        end = time.time()
        for i, (input, smap, fixmap, info) in enumerate(val_loader):
            # compute output
            if config.TEST.FLIP_TEST:
                flip_input = input.flip(3)
                flip_outputs = model(flip_input)
                outputs = (model(input) + flip_outputs.flip(3)) / 2
            else:
                outputs = model(input)

            smap = smap.cuda(non_blocking=True)
            fixmap = fixmap.cuda(non_blocking=True)

            if config.DATASET.SAMPLER == "RandomIdentitySampler":
                outputs = outputs[0]

            loss, other_info = criterion(outputs, smap, fixmap)

            if config.TRAIN.OHEM != -1:
                loss = loss.mean()

            num_images = len(input) if isinstance(input, list) else input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if returnRes:
                out = outputs if not isinstance(outputs, list) else outputs[0]
                # if isinstance(outputs, list):
                #     out = (outputs[0] + F.upsample(outputs[1], size=outputs[0].size()[-2:], mode='bilinear', align_corners=True)) / 2
                for idx,sal_map in enumerate(out):
                    res.append({'image_id':info['id'][idx].item(), 'saliency_map':(sal_map/torch.max(sal_map)).cpu().numpy().squeeze()})

            if i % config.PRINT_FREQ == 0 or i == len(val_loader)-1:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                if other_info is not None:
                    msg += other_info
                logger.info(msg)

                # prefix = '{}_{}'.format(
                #     os.path.join(output_dir, 'val'), i
                # )
                # save_debug_images(config, input, meta, target, pred*4, output,
                #                   prefix)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer_dict['valid_global_steps'] = global_steps + 1

    perf_indicator = losses.avg
    is_larger_better = False

    if returnRes:
        return perf_indicator, res
    else:
        return perf_indicator, is_larger_better


def test(config, test_loader, model, output_dir):
    from tqdm import tqdm
    import cv2
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for input, info in tqdm(test_loader):
            # compute output
            if config.TEST.FLIP_TEST:
                flip_input = input.flip(3)
                flip_outputs = model(flip_input)
                outputs = (model(input) + flip_outputs.flip(3)) / 2
            else:
                outputs = model(input)
            if config.DATASET.SAMPLER == "RandomIdentitySampler":
                info['size'] = np.array(info['size']).transpose((2,0,1)).reshape(2,-1)
                info['name'] = np.array(info['name']).transpose().reshape(-1)
                vector, out = outputs
                vector = vector.view(-1)
            else:
                out = outputs if not isinstance(outputs, list) else outputs[0]
            for idx,sal_map in enumerate(out):
                if config.DATASET.SAMPLER == "RandomIdentitySampler":
                    sal_map = (sal_map/torch.max(sal_map)).cpu().numpy().squeeze()
                else:
                    sal_map = (sal_map/torch.max(sal_map)).cpu().numpy().squeeze()
                if config.TEST.SAME_SIZE:
                    # if config.DATASET.SAMPLER == "RandomIdentitySampler":
                    #     size = (int(info['size'][0][idx]), int(info['size'][1][idx]))
                    # else:
                    #     size = (info['size'][0][idx].item(), info['size'][1][idx].item())
                    size = (int(info['size'][0][idx]), int(info['size'][1][idx]))
                    sal_map = rebuild_map(sal_map, size)
                    sal_map = cv2.resize(sal_map, (size[1],size[0]))
                cv2.imwrite(output_dir+'/'+info['name'][idx]+config.TEST.SUFFIX, sal_map*255)
                # cv2.imwrite(output_dir+'/'+'img_{}.png'.format(idx), (input.cpu().numpy()[idx].transpose((1,2,0))*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406])*255)

def rebuild_map(sal_map, size, name=None):
    shape_r, shape_c = sal_map.shape
    rows_rate = size[0]/shape_r
    cols_rate = size[1]/shape_c
    if rows_rate > cols_rate:
        new_rows, new_cols = shape_r, int(shape_r * size[1] / size[0])
        sal_map = sal_map[:, ((shape_c - new_cols) // 2):((shape_c - new_cols) // 2 + new_cols)]
    else:
        new_rows, new_cols = int(shape_c * size[0] / size[1]), shape_c
        sal_map = sal_map[((shape_r - new_rows) // 2):((shape_r - new_rows) // 2 + new_rows), :]
    return sal_map


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
