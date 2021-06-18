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

@DATASETS.register_module()
class hyper(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('noncancer', 'cancer')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super(hyper, self).__init__(
            img_suffix='.hdr', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


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

        con_mat = np.zeros((2,2))
        for result, gt in zip(results, gt_seg_maps):
            con_mat += metrics.confusion_matrix(gt.flatten(), result.flatten(), labels=[1, 0])
        print_log('accuracy:{}'.format(accuracy(con_mat)), logger=logger)
        print_log('kappa:{}'.format(kappa(con_mat)), logger=logger)
        print_log('mIoU:{}'.format(ret_metrics_mean[2]), logger=logger)
        print_log('mDice:{}'.format(ret_metrics_mean[3]), logger=logger)
        # print_log('precision:{}'.format(precision(con_mat)), logger=logger)
        # print_log('sensitivity:{}'.format(sensitivity(con_mat)), logger=logger)
        # print_log('specificity:{}'.format(specificity(con_mat)), logger=logger)

        return eval_results


def kappa(matrix):
    matrix = np.array(matrix)
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)
def sensitivity(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[1][0])
def specificity(matrix):
    return matrix[1][1]/(matrix[1][1]+matrix[0][1])
def precision(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[0][1])
def accuracy(matrix):
    return (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])