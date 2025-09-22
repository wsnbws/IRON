from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import os.path as osp


@DATASETS.register_module()
class DrivableData(CustomDataset):
    CLASSES = (
    "_background_",
    "drivable_area")
    
    PALETTE = [[0, 0, 0], [0, 255, 0],]
    
    def __init__(self, **kwargs):
        super(DrivableData, self).__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)