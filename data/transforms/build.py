# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])
    # normalizer    = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    # if is_train:
    #     transform = T.Compose([
    #         T.Resize(cfg.INPUT.SIZE_TRAIN),
    #         T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
    #         T.Pad(cfg.INPUT.PADDING),
    #         T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
    #         T.ToTensor(),
    #         normalizer,
    #         RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    #     ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            #normalize_transform
            normalizer
        ])

    return transform
