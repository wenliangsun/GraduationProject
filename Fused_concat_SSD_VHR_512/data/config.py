# config.py
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home, "/media/sunwl/Datum/Datasets/PASCAL_VOC/VOCdevkit2012")

# note: if you used our download scripts, this should be right
VOCroot = ddir  # path to VOCdevkit root dir
VHRroot = "/media/sunwl/Datum/Datasets/NWPU VHR-10 dataset"
ROSDroot = "/media/sunwl/Datum/Datasets/ROSD_dataset/ROSD"

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4

# SSD300 CONFIGS
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name': 'v2',
}

# use average pooling layer as last layer before multibox layers
v1 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'aspect_ratios': [[1, 1, 2, 1 / 2], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3],
                      [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3]],

    'variance': [0.1, 0.2],

    'clip': True,

    'name': 'v1',
}
