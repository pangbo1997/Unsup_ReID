# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, _, _, _, mem_idx= zip(*batch)
    mem_idx = torch.tensor(mem_idx, dtype=torch.int64)
    return torch.stack(imgs, dim=0), mem_idx


def val_collate_fn(batch):
    imgs, pids, camids, _ ,_= zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids
