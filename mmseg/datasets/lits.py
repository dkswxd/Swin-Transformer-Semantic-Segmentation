import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset

from functools import reduce
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from mmseg.core import eval_metrics
import os
from sklearn import metrics
from mmseg.utils import get_root_logger
import SimpleITK as sitk

@DATASETS.register_module()
class LiTS(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'liver', 'turmor')

    PALETTE = [[0, 0, 0], [0, 255, 0], [0, 0, 255]]

    def __init__(self, split, replace=('volume', 'segmentation'), **kwargs):
        self.replace=replace
        self.transpose = 'dhw2hwd'
        super(LiTS, self).__init__(
            img_suffix='.nii', seg_map_suffix='.nii', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        with open(split) as f:
            for line in f:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name.replace(*self.replace) + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = sitk.ReadImage(seg_map, imageIO="NiftiImageIO")
                gt_seg_map = sitk.GetArrayFromImage(gt_seg_map)
                if self.transpose == 'dhw2hwd':
                    gt_seg_map = np.transpose(gt_seg_map, (1,2,0))
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps