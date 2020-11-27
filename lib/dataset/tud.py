from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
import pickle
import torch
import torch.utils.data as data

from pycocotools.coco import COCO

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.transforms import hoe_heatmap_gen

logger = logging.getLogger(__name__)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

class tud_dataset(data.Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.root = root
        self.is_train = is_train
        if is_train:
            db_path_1 = os.path.join(self.root, 'annot', 'train_tud')
            db_1 = load_obj(db_path_1)
            db_path_2 = os.path.join(self.root, 'annot', 'val_tud')
            db_2 = load_obj(db_path_2)
            self.tud_db = db_1 + db_2
        else:
            db_path = os.path.join(self.root, 'annot', 'test_tud')
            self.tud_db = load_obj(db_path)
        self.flip = cfg.DATASET.FLIP
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.hoe_sigma = cfg.DATASET.HOE_SIGMA
        self.transform = transform
    def _box2cs(self, box):
        x1, y1, x2, y2 = box[:]
        center = np.zeros((2), dtype=np.float32)
        center[0] = (x1 + x2) /2
        center[1] = (y1 + y2) /2
        w = x2 - x1
        h = y2 - y1
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale

    def load_db(self, index):
        imgfile= self.tud_db[index]['image']
        bbox = self.tud_db[index]['bbox']
        degree = self.tud_db[index]['degree']
        degree = degree - 90
        if degree < 0:
            degree += 360
        c,s = self._box2cs(bbox)
        return imgfile, c, s, degree

    def __len__(self,):
        return len(self.tud_db)

    def __getitem__(self, index):
        imgfile, c,s, degree = self.load_db(index)
        if self.is_train:
            imgfile = os.path.join(self.root, 'images', imgfile)
        else:
            imgfile = os.path.join(self.root, 'images', imgfile)
        data_numpy = cv2.imread(imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(imgfile))
            raise ValueError('Fail to read {}'.format(imgfile))

        if self.is_train:
            sf = self.scale_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                c[0] = data_numpy.shape[1] - c[0] - 1

                degree = 360 - degree
        value_degree = degree
        degree = round(degree/5.0)
        if degree == 72:
            degree = 0
        degree = hoe_heatmap_gen(degree, 72, sigma=self.hoe_sigma)
        trans = get_affine_transform(c, s, 0, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        if self.transform:
            input = self.transform(input)
        input = input.float()

        meta = {
            'image_path': imgfile,
            'center': c,
            'scale': s,
            'val_dgree': value_degree,
        }
        return input, 0, 0, degree, meta

if __name__ == '__main__':
    import argparse
    from config import cfg
    from config import update_config
    import torchvision.transforms as transforms
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cfg = "experiments/tud/lrle-3.yaml"
    args.opts, args.modelDir, args.logDir, args.dataDir = "", "", "", ""
    update_config(cfg, args)
    normalize = transforms.Normalize(
        mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]
    )
    train_dataset = tud_dataset(
        cfg, cfg.DATASET.TRAIN_ROOT, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    for i, b in enumerate(train_loader):
        if i == 3:
            break
        else:
            img = b[0][0]
            img = img.numpy()
            img_tmp = np.transpose(img, axes=[1, 2, 0])
            img_tmp *= [0.229, 0.229, 0.229]
            img_tmp += [0.485, 0.485, 0.485]
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(212)
            plt.imshow(img_tmp)
    plt.show(block=True)

