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

    # CLASSES = ('background', 'liver', 'turmor')
    CLASSES = ('background', 'liver')

    PALETTE = [[0, 0, 0], [0, 255, 0], [0, 0, 255]]

    def __init__(self, split, replace=('volume', 'segmentation'), **kwargs):
        self.replace=replace
        self.transpose = 'dhw2hwd'
        self.lits_remove_turmor = True
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
                if self.lits_remove_turmor:
                    gt_seg_map[gt_seg_map > 0] = 1
                if self.transpose == 'dhw2hwd':
                    gt_seg_map = np.transpose(gt_seg_map, (1,2,0))
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def format_results(self, results, **kwargs):
        assert len(results) == len(self.img_infos)
        for result, img_info in zip(results, self.img_infos):

            result = sitk.GetImageFromArray(result.astype(np.uint8))
            sitk.WriteImage(result, os.path.join('./LiTS_val/results',img_info['filename'].replace(*self.replace)))


    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        results_sliced = []
        gt_seg_maps_sliced = []
        for _results, _gt_seg_maps in zip(results, gt_seg_maps):
            assert _results.shape == _gt_seg_maps.shape
            for _slice in range(_results.shape[-1]):
                results_sliced.append(_results[:,:,_slice])
                gt_seg_maps_sliced.append(_gt_seg_maps[:,:,_slice])
        results = results_sliced
        gt_seg_maps = gt_seg_maps_sliced
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
