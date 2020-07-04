import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm
import os
import os.path as osp
import glob
import random
from torchvision import transforms
# from collections import namedtuple
import copy
import scipy.io as sio
# import pandas as pd
# import multiprocessing
import cv2
import logging
from PIL import Image

# from utils.transforms import get_affine_transform

from dataset.salicon_evaluation.salicontool.salicon import SALICON
from dataset.salicon_evaluation.saliconeval.eval import SALICONEval

logger = logging.getLogger(__name__)

def load_and_resize(img_path, loader, size):
    img = loader(img_path).resize(size,Image.LANCZOS)
    return img

def get_all_file_dir(root_list, pattern):
    if not isinstance(root_list, list):
        root_list = [root_list]
    file_dir = []
    for root in root_list:
        file_dir += glob.glob(osp.join(root, pattern))
    return file_dir

def default_loader(img_path):
    img = None
    with Image.open(img_path) as temp_img:
        img = temp_img.convert('RGB')
    return img

def padding(img, shape_r=480, shape_c=640, channels=3, pad_type=""):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if pad_type == 'roll':
        assert rows_rate > cols_rate

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
        if pad_type == 'roll':
            img_padded[:, :((img_padded.shape[1] - new_cols) // 2)] = img[:,-((img_padded.shape[1] - new_cols) // 2):]
            img_padded[:, ((img_padded.shape[1] - new_cols) // 2 + new_cols):] = img[:,:shape_c-((img_padded.shape[1] - new_cols) // 2 + new_cols)]
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


def preprocess_image(path, shape_r, shape_c, pad_type=""):
    original_image = cv2.imread(path)
    padded_image = padding(original_image, shape_r, shape_c, 3, pad_type)
    ims = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    return ims.squeeze()

def preprocess_map(path, shape_r, shape_c, pad_type=""):
    original_map = cv2.imread(path, 0)
    padded_map = padding(original_map, shape_r, shape_c, 1, pad_type)
    ims = padded_map.astype(np.float32)
    ims /= 255.
    return ims.squeeze()

def preprocess_fix(path, shape_r, shape_c):
    mat = sio.loadmat(path)
    gazes = mat["gaze"]
    coords = []
    for gaze in gazes:
        coords.extend(gaze[0][2])
    coords = np.array(coords)[:,::-1]
    raw_shape_r, raw_shape_c = mat["resolution"][0]
    rate_r = shape_r * 1.0 / raw_shape_r
    rate_c = shape_c * 1.0 / raw_shape_c
    coords[:,0] = coords[:,0] * rate_r
    coords[:,1] = coords[:,1] * rate_c
    coords = coords.astype(np.uint16)
    return coords

def generate_fixmaps(coords, shape_r, shape_c):
    ims = np.zeros((shape_r, shape_c))
    for coord in coords:
        if coord[0] >= 0 and coord[0] < shape_r and coord[1] >= 0 and coord[1] < shape_c:
            ims[coord[0], coord[1]] = 1.0
    ims = ims.astype(np.float32)
    return ims


class SALICON_Dataset(Dataset):
    """docstring for SALICON_Dataset"""
    def __init__(self, cfg, data_root_list, image_set, is_train, transform=None):
        super(SALICON_Dataset, self).__init__()
        self.cfg= cfg
        self.image_size= np.array(cfg.MODEL.IMAGE_SIZE)
        self.map_size= np.array(cfg.MODEL.HEATMAP_SIZE)
        self.data_root_list= data_root_list
        self.transform = transform
        self.loader = default_loader
        self.image_set = image_set
        self.return_info = True
        self.is_train = is_train
        self.flip = cfg.DATASET.FLIP
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR

        # assert ('17' in data_root_list)

        if is_train:
            if image_set not in ['train','all']:
                error_msg = "image_set({}) is not in ['train','all']".format(image_set)
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            if image_set not in ['val','test','smalltest','train']:
                error_msg = "image_set({}) is not in ['train','all']".format(image_set)
                logger.error(error_msg)
                raise ValueError(error_msg)

        pattern = '*.jpg'
        self.images = get_all_file_dir(osp.join(self.data_root_list,'images',image_set), pattern)
        # if image_set != "test" and 'local' in self.data_root_list:
        #     self.images = [x for x in self.images if not x.endswith('1.jpg') and not x.endswith('6.jpg')]
        self.images = np.sort(self.images)

        pattern = '*.png'
        self.maps = get_all_file_dir(osp.join(self.data_root_list,'maps',image_set), pattern)
        # if image_set != "test" and 'local' in self.data_root_list:
        #     self.maps = [x for x in self.maps if not x.endswith('1.png') and not x.endswith('6.png')]
        self.maps = np.sort(self.maps)
        self.fixs = np.array(get_all_file_dir(osp.join(self.data_root_list,'fixations',image_set), '*.mat'))
        self.fixs = np.sort(self.fixs)

        if cfg.DATASET.COLORJITTER:
            self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0)

        if len(self.images) <= 0:
            error_msg = "len(self.images) = {} is not allowed. Check the data_root_path: {}".format(len(self.images), data_root_list)
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not (image_set=='smalltest' or image_set=='test' or len(self.images)==len(self.maps)):
            error_msg = "len(self.images) = {:d}, but len(self.maps) = {:d}".format(len(self.images), len(self.maps))
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.entries_num = len(self.images)

    def getitem(self,index):
        image = preprocess_image(self.images[index], self.image_size[0], self.image_size[1], pad_type=self.cfg.DATASET.PAD_TYPE)
        if len(self.maps) != 0:
            smap = preprocess_map(self.maps[index], self.map_size[0], self.map_size[1], pad_type=self.cfg.DATASET.PAD_TYPE)
            if len(self.fixs) > 0:
                fix = preprocess_fix(self.fixs[index], self.map_size[0], self.map_size[1], pad_type=self.cfg.DATASET.PAD_TYPE)

        if self.is_train:
            if self.cfg.DATASET.COLORJITTER:
                image = np.array(self.colorjitter(Image.fromarray(image)))

            if self.flip and random.random() <= 0.5:
                image = image[:, ::-1, :].copy()
                smap = smap[:, ::-1].copy()
                if len(self.fixs) > 0:
                    fix[:,1] = self.map_size[1] - 1 - fix[:,1]

            if self.scale_factor > 0 and random.random() <= 0.5:
                sf = self.scale_factor
                sf = np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
                image_tmp = cv2.resize(image, (int(self.image_size[1]*sf), int(self.image_size[0]*sf)), interpolation=cv2.INTER_AREA)
                smap_tmp = cv2.resize(smap, (int(self.map_size[1]*sf), int(self.map_size[0]*sf)), interpolation=cv2.INTER_AREA)

                if len(self.fixs) > 0:
                    fix = fix * sf

                if sf > 1:
                    image = image_tmp[((image_tmp.shape[0]-image.shape[0])//2):((image_tmp.shape[0]-image.shape[0])//2+image.shape[0]), \
                            ((image_tmp.shape[1]-image.shape[1])//2):((image_tmp.shape[1]-image.shape[1])//2+image.shape[1])]
                    smap = smap_tmp[((smap_tmp.shape[0]-smap.shape[0])//2):((smap_tmp.shape[0]-smap.shape[0])//2+smap.shape[0]), \
                            ((smap_tmp.shape[1]-smap.shape[1])//2):((smap_tmp.shape[1]-smap.shape[1])//2+smap.shape[1])]
                    if len(self.fixs) > 0:
                        fix[:,0] = fix[:,0] - (smap_tmp.shape[0]-smap.shape[0])//2
                        fix[:,1] = fix[:,1] - (smap_tmp.shape[1]-smap.shape[1])//2
                        fix = fix.astype(np.uint16)
                else:
                    image = np.zeros(image.shape, dtype=np.uint8)
                    smap = np.zeros(smap.shape, dtype=np.float32)
                    image[((image.shape[0]-image_tmp.shape[0])//2):((image.shape[0]-image_tmp.shape[0])//2+image_tmp.shape[0]), \
                            ((image.shape[1]-image_tmp.shape[1])//2):((image.shape[1]-image_tmp.shape[1])//2+image_tmp.shape[1])] = image_tmp
                    smap[((smap.shape[0]-smap_tmp.shape[0])//2):((smap.shape[0]-smap_tmp.shape[0])//2+smap_tmp.shape[0]), \
                            ((smap.shape[1]-smap_tmp.shape[1])//2):((smap.shape[1]-smap_tmp.shape[1])//2+smap_tmp.shape[1])] = smap_tmp
                    if len(self.fixs) > 0:
                        fix[:,0] = fix[:,0] + (smap.shape[0]-smap_tmp.shape[0])//2
                        fix[:,1] = fix[:,1] + (smap.shape[1]-smap_tmp.shape[1])//2
                        fix = fix.astype(np.uint16)


            if self.rotation_factor > 0 and random.random() <= 0.5:
                rf = self.rotation_factor
                rf = np.clip(np.random.randn()*rf, -rf, rf)
                M = cv2.getRotationMatrix2D((self.image_size[1] // 2, self.image_size[0] // 2), int(rf), 1)
                image = cv2.warpAffine(image, M, (self.image_size[1], self.image_size[0]))
                M = cv2.getRotationMatrix2D((self.map_size[1] // 2, self.map_size[0] // 2), int(rf), 1)
                smap = cv2.warpAffine(smap, M, (self.map_size[1], self.map_size[0]))

                if len(self.fixs) > 0:
                    M = cv2.getRotationMatrix2D((self.map_size[1] // 2, self.map_size[0] // 2), -int(rf), 1)
                    ones = np.ones(shape=(len(fix), 1))
                    fix_ones = np.hstack([fix, ones])
                    fix = M.dot(fix_ones.T).T.astype(np.uint16)

        if self.transform:
            image = self.transform(image)

        return_list = [image]
        if len(self.maps) != 0:
            return_list.append(smap.copy())
            if len(self.fixs) > 0:
                fix_map = generate_fixmaps(fix, self.map_size[0], self.map_size[1])
                return_list.append(fix_map.copy())
            else:
                return_list.append(torch.zeros(1))

        if self.return_info:
            info = dict()
            info['name'] = self.images[index].split('/')[-1].split('.')[0]
            info['size'] = [str(s) for s in cv2.imread(self.images[index]).shape[:2]]
            # if 'MIT1003' in self.data_root_list or 'cat2000' in self.data_root_list:
            #     info['id'] = info['name']
            # else:
            #     info['id'] = int(info['name'].split('_')[-1])
            return_list.append(info)

        return tuple(return_list)

    def __getitem__(self,index):
        if not isinstance(index, (tuple, list)):
            return self.getitem(index)
        else:
            imgs = []
            size_info = []
            name_info = []
            if len(self.maps) != 0:
                target = []
            for idx in index:
                if self.return_info:
                    if len(self.maps) != 0:
                        img, smap, fix, info = self.getitem(idx)
                    else:
                        img, info = self.getitem(idx)
                    size_info.append(info['size'])
                    name_info.append(info['name'])
                else:
                    if len(self.maps) != 0:
                        img, smap, fix = self.getitem(idx)
                    else:
                        img = self.getitem(idx)
                imgs.append(img)
                if len(self.maps) != 0:
                    target.append(smap.max())
            return_list = [imgs]
            if len(self.maps) != 0:
                return_list.append(np.array(target))
                return_list.append(fix)
            if self.return_info:
                info['idx_list'] = index
                info['size'] = size_info
                info['name'] = name_info
                return_list.append(info)
            return tuple(return_list)


    def __len__(self):
        return self.entries_num

    def evaluate(self, output_dir, predRes, jsonName='', annoFile=None):
        if annoFile is None:
            annoFile = self.cfg.TEST.ANNO_FILE

        if not (len(annoFile)>0 and osp.exists(annoFile)):
            error_msg = "annoFile({}) should be a valid file path".format(annoFile)
            logger.error(error_msg)
            raise ValueError(error_msg)

        assert '17' in annoFile

        saliconAnno = SALICON(annoFile)
        saliconRes = saliconAnno.loadRes(predRes)
        saliconEval = SALICONEval(saliconAnno, saliconRes)
        saliconEval.params['image_id'] = saliconRes.getImgIds()

        saliconEval.evaluate(filterAnns=False)

        logger.info("=> Final Result for each Metric:")
        for metric, score in saliconEval.eval.items():
            logger.info('\t%s: %.3f'%(metric, score))

        import json

        # save evaluation results to ./results folder
        json_file = osp.join(output_dir, 'evalImgsFile_%s.json'%(jsonName))
        logger.info("=> dump josn file: {}".format(json_file))
        json.dump(saliconEval.evalImgs, open(json_file, 'w'))

        json_file = osp.join(output_dir, 'evalFile_%s.json'%(jsonName))
        logger.info("=> dump josn file: {}".format(json_file))
        json.dump(saliconEval.eval, open(json_file, 'w'))



if __name__ == '__main__':
    pass