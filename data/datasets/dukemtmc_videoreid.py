# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import re
import urllib
import zipfile

import os.path as osp

from utils.iotools import mkdir_if_missing
from .bases import BaseVideoDataset
import os

class DukeMTMCVideoreID(BaseVideoDataset):


    dataset_dir = 'DukeMTMC-VideoReID'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(DukeMTMCVideoreID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._download_data()
        self._check_before_run()

        train,train_camids = self._process_dir(self.train_dir, relabel=False)
        query,_ = self._process_dir(self.query_dir, relabel=False)
        gallery,_ = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> DukeMTMCVideo-reID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.train_camids = train_camids
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_videodata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_videodata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_videodata_info(self.gallery)

    def _download_data(self):
        if osp.exists(self.dataset_dir):
            print("This dataset has been downloaded.")
            return

        print("Creating directory {}".format(self.dataset_dir))
        mkdir_if_missing(self.dataset_dir)
        fpath = osp.join(self.dataset_dir, osp.basename(self.dataset_url))

        print("Downloading DukeMTMC-reID dataset")
        urllib.request.urlretrieve(self.dataset_url, fpath)

        print("Extracting files")
        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(self.dataset_dir)
        zip_ref.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        dataset=[]
        camids = []
        person_list = os.listdir(os.path.join(dir_path));
        person_list.sort()
        for person_id in person_list:
            count = 0
            pid = int(person_id)
            videos = os.listdir(os.path.join(dir_path, person_id));
            videos.sort()
            for video_id in videos:
                video_path = os.path.join(dir_path, person_id, video_id)
                fnames = os.listdir(video_path)
                frame_list = []
                for fname in fnames:
                    count += 1
                    cam = int(fname[6]) - 1
                    assert 0 <= pid <= 7140
                    assert 0 <= cam <= 8
                    frame_list.append(osp.join(video_path,fname))
                dataset.append((frame_list,pid,cam))
                camids.append(cam)
        return dataset,camids
