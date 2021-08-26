import os.path as osp

import mmcv
import numpy as np
import cv2

from ..builder import PIPELINES
import SimpleITK as sitk

@PIPELINES.register_module()
class NiiLoadImageFromFile(object):
    """Load an image from file.
    """

    def __init__(self, transpose='dhw2hwd'):
        self.transpose = transpose

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = sitk.ReadImage(filename, imageIO="NiftiImageIO")
        spacing = (1, 1, img.GetSpacing()[2])
        img = sitk.GetArrayFromImage(img)
        if self.transpose == 'dhw2hwd':
            img = np.transpose(img, (1,2,0))

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img.astype(np.float32)
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['spacing'] = spacing
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class NiiLoadAnnotationsFromFile(object):
    """Load annotations for semantic segmentation.
    """

    def __init__(self,reduce_zero_label=False, transpose='dhw2hwd',lits_remove_turmor=False):
        self.reduce_zero_label = reduce_zero_label
        self.transpose = transpose
        self.lits_remove_turmor = lits_remove_turmor


    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']

        gt_semantic_seg = sitk.ReadImage(filename, imageIO="NiftiImageIO")
        # assert results['spacing'] == gt_semantic_seg.GetSpacing()
        gt_semantic_seg = sitk.GetArrayFromImage(gt_semantic_seg).astype(np.uint8)
        if self.lits_remove_turmor:
            gt_semantic_seg[gt_semantic_seg > 0] = 1
        if self.transpose == 'dhw2hwd':
            gt_semantic_seg = np.transpose(gt_semantic_seg, (1,2,0))

        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id

        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label})'
        return repr_str
