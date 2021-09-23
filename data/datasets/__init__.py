# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .dukemtmc_videoreid import DukeMTMCVideoreID
from .dataset_loader import ImageDataset,VideoDataset
from .mars import Mars
from .msmt17 import MSMT17
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'dukemtmc-video':DukeMTMCVideoreID,
    'mars':Mars,
    'msmt17':MSMT17
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
