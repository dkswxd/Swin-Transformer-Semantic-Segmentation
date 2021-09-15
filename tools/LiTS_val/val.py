"""

测试脚本
"""

import os
import copy
import collections
from time import time

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology

from calculate_metrics import Metirc

# 为了计算dice_global定义的两个变量
dice_intersection = 0.0  
dice_union = 0.0

file_name = []  # 文件名称
time_pre_case = []  # 单例数据消耗时间

# 定义评价指标
liver_score = collections.OrderedDict()
liver_score['dice'] = []
liver_score['jacard'] = []
liver_score['voe'] = []
liver_score['fnr'] = []
liver_score['fpr'] = []
liver_score['assd'] = []
liver_score['rmsd'] = []
liver_score['msd'] = []

# 定义网络并加载参数

for file_index, file in enumerate(os.listdir('./results/')):

    file_name.append(file)
    ct = sitk.ReadImage(os.path.join('../../data/LiTS/Training/', file.replace('segmentation','volume')), sitk.sitkUInt8)
    # 将pred读入内存
    pred_seg = sitk.ReadImage(os.path.join('./results/', file), sitk.sitkUInt8)
    pred_seg = sitk.GetArrayFromImage(pred_seg)
    pred_seg = np.transpose(pred_seg,(2,0,1))
    pred_seg[pred_seg > 0] = 1

    # 将金标准读入内存
    seg = sitk.ReadImage(os.path.join('../../data/LiTS/Training/', file), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    # 对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充
    liver_seg = copy.deepcopy(pred_seg)
    liver_seg = measure.label(liver_seg, connectivity=1)
    props = measure.regionprops(liver_seg)

    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index

    liver_seg[liver_seg != max_index] = 0
    liver_seg[liver_seg == max_index] = 1

    liver_seg = liver_seg.astype(np.bool_)
    morphology.remove_small_holes(liver_seg, 5e4, connectivity=2, in_place=True)
    liver_seg = liver_seg.astype(np.uint8)

    # 计算分割评价指标
    liver_metric = Metirc(seg_array, liver_seg, ct.GetSpacing())

    liver_score['dice'].append(liver_metric.get_dice_coefficient()[0])
    liver_score['jacard'].append(liver_metric.get_jaccard_index())
    liver_score['voe'].append(liver_metric.get_VOE())
    liver_score['fnr'].append(liver_metric.get_FNR())
    liver_score['fpr'].append(liver_metric.get_FPR())
    liver_score['assd'].append(liver_metric.get_ASSD())
    liver_score['rmsd'].append(liver_metric.get_RMSD())
    liver_score['msd'].append(liver_metric.get_MSD())

    dice_intersection += liver_metric.get_dice_coefficient()[1]
    dice_union += liver_metric.get_dice_coefficient()[2]



# 将评价指标写入到exel中
liver_data = pd.DataFrame(liver_score, index=file_name)

liver_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(liver_data.columns))
liver_statistics.loc['mean'] = liver_data.mean()
liver_statistics.loc['std'] = liver_data.std()
liver_statistics.loc['min'] = liver_data.min()
liver_statistics.loc['max'] = liver_data.max()

writer = pd.ExcelWriter('./result.xlsx')
liver_data.to_excel(writer, 'liver')
liver_statistics.to_excel(writer, 'liver_statistics')
writer.save()

# 打印dice global
print('dice global:', dice_intersection / dice_union)
