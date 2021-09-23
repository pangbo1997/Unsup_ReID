# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
#    print(img_path)
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, camids,transform=None):
        self.dataset = dataset
        self.camids=camids
        self.transform = transform

    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path,index

class VideoDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, camids,transform=None,num_samples=16, is_training=True, max_frames= 900):
        self.dataset = dataset
        self.camids=camids
        self.transform = transform
        self.num_samples=4

        self.is_training=is_training
        self.max_frames=max_frames
    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, index):
        frames_path, pid, camid = self.dataset[index]
        if self.is_training:
            if len(frames_path) >= self.num_samples:
                images_path = random.sample(frames_path, self.num_samples)
            else:
                images_path = random.choices(frames_path,  k=self.num_samples)
            images_path.sort()
        else: # for evaluation, we use all the frames
            if len(frames_path) > 900:  # to avoid the insufficient memory
                images_path = random.sample(frames_path, 900)
            else:
                images_path=frames_path

        imgs=[]
        for img_path in images_path:
            img = read_image(img_path)

            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs=torch.stack(imgs, dim=0)
        return imgs, pid, camid, images_path,index
