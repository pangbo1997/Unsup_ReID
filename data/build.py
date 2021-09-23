# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset,VideoDataset
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset= init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    if cfg.DATASETS.IS_VIDEO:
        num_classes = dataset.num_train_pids
        train_set = VideoDataset(dataset.train, dataset.train_camids, train_transforms,is_training=True)
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )

        val_set = VideoDataset(dataset.query + dataset.gallery, None, val_transforms,is_training=False)
        val_loader = DataLoader(
            val_set, batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        return train_loader, val_loader, len(dataset.query), num_classes
    else:
        num_classes = dataset.num_train_pids
        train_set = ImageDataset(dataset.train,dataset.train_camids, train_transforms)
        train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn
            )


        val_set = ImageDataset(dataset.query + dataset.gallery, None,val_transforms)
        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        return train_loader, val_loader, len(dataset.query), num_classes
