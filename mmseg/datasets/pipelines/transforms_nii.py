import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
import torch.nn.functional as F
from ..builder import PIPELINES
import torch

@PIPELINES.register_module()
class NiiSpacingNormalize(object):
    def __init__(self, target_spacing=(1, 1, 1)):
        self.target_spacing = target_spacing


    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        with torch.no_grad():
            img = torch.from_numpy(results['img']).unsqueeze(0).unsqueeze(0)
            img = F.interpolate(input=img, scale_factor=results['spacing'], mode='trilinear',
                                       align_corners=False, recompute_scale_factor=False)
            img = img.numpy().squeeze(0).squeeze(0)
            results['img'] = img

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        with torch.no_grad():
            for key in results.get('seg_fields', []):
                gt_seg = torch.from_numpy(results[key]).unsqueeze(0).unsqueeze(0)
                gt_seg = F.interpolate(input=gt_seg, scale_factor=results['spacing'], mode='nearest',
                                       recompute_scale_factor=False)
                gt_seg = gt_seg.numpy().squeeze(0).squeeze(0)
                results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'target_spacing={self.target_spacing},'
        return repr_str

@PIPELINES.register_module()
class NiiResize(object):
    """Resize images & seg.

    Args:
        ratio_range (tuple[float]): (min_ratio, max_ratio)
    """

    def __init__(self, ratio_range=None):
        if ratio_range is not None:
            self.min_ratio, self.max_ratio = ratio_range

    def _random_scale(self, results):
        results['scale'] = np.random.random_sample() * (self.max_ratio - self.min_ratio) + self.min_ratio

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        with torch.no_grad():
            img = results['img']
            h, w, d = img.shape[:3]
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            img = F.interpolate(input=img, scale_factor=results['scale'], mode='trilinear',
                                       align_corners=False, recompute_scale_factor=False)
            img = img.numpy().squeeze(0).squeeze(0)
            new_h, new_w, new_d = img.shape[:3]
            w_scale = new_w / w
            h_scale = new_h / h
            d_scale = new_d / d

            scale_factor = np.array([w_scale, h_scale, d_scale, w_scale, h_scale, d_scale],
                                    dtype=np.float32)
            results['img'] = img
            results['img_shape'] = img.shape
            results['pad_shape'] = img.shape  # in case that there is no padding
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = True

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        with torch.no_grad():
            for key in results.get('seg_fields', []):
                gt_seg = torch.from_numpy(results[key]).unsqueeze(0).unsqueeze(0)
                gt_seg = F.interpolate(input=gt_seg, scale_factor=results['scale'], mode='nearest',
                                       recompute_scale_factor=False)
                gt_seg = gt_seg.numpy().squeeze(0).squeeze(0)
                results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'min_ratio={self.min_ratio},'
                     f'max_ratio={self.max_ratio},')
        return repr_str


@PIPELINES.register_module()
class NiiRandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(tuple[str], optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction=None):
        self.prob = prob
        self.direction = direction
        assert direction in ('horizontal', 'vertical', 'depth', None)
        if prob is not None:
            assert prob >= 0 and prob <= 1

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = np.random.rand() < self.prob
            results['flip'] = flip
        if 'flip_direction' not in results:
            if self.direction == None:
                results['flip_direction'] = random.choice(('horizontal', 'vertical', 'depth'))
            else:
                results['flip_direction'] = self.direction
        if results['flip']:
            if results['flip_direction'] == 'horizontal':
                results['img'] = np.ascontiguousarray(results['img'][::-1, :, :])
                for key in results.get('seg_fields', []):
                    results[key] = np.ascontiguousarray(results[key][::-1, :, :])
            if results['flip_direction'] == 'vertical':
                results['img'] = np.ascontiguousarray(results['img'][:, ::-1, :])
                for key in results.get('seg_fields', []):
                    results[key] = np.ascontiguousarray(results[key][:, ::-1, :])
            if results['flip_direction'] == 'depth':
                results['img'] = np.ascontiguousarray(results['img'][:, :, ::-1])
                for key in results.get('seg_fields', []):
                    results[key] = np.ascontiguousarray(results[key][:, :, ::-1])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class NiiPad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def impad(self,
              img,
              shape=None,
              padding=None,
              pad_val=0,
              padding_mode='constant'):
        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            padding = [0, shape[2] - img.shape[2], 0, shape[1] - img.shape[1], 0, shape[0] - img.shape[0]]

        # check padding
        if (isinstance(padding, tuple) or isinstance(padding, list)) and len(padding) in [2, 6]:
            if len(padding) == 2:
                padding = [padding[0], padding[1], padding[0], padding[1], padding[0], padding[1]]
        else:
            raise ValueError('Padding must be a int or a 2, or 6 element tuple.'
                             f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant']

        with torch.no_grad():
            img = F.pad(torch.from_numpy(img), padding, mode=padding_mode, value=pad_val).numpy()
        return img

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = self.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            pad_h = int(np.ceil(results['img'].shape[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(results['img'].shape[1] / self.size_divisor)) * self.size_divisor
            pad_d = int(np.ceil(results['img'].shape[2] / self.size_divisor)) * self.size_divisor
            padded_img = self.impad(
                results['img'], shape=(pad_h, pad_w, pad_d), pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = self.impad(
                results[key],
                shape=results['pad_shape'][:3],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NiiNormalizeImage(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self):
        pass

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = results['img'] - results['img'].mean()
        results['img'] = results['img'] / results['img'].std()
        results['img_norm_cfg'] = dict(mean=[128], std=[16])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class NiiClipImageValue(object):
    def __init__(self, min_value=-200,max_value=250):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, results):
        results['img'] = np.clip(results['img'], a_min=self.min_value, a_max=self.max_value)
        return results


@PIPELINES.register_module()
class NiiRandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        margin_d = max(img.shape[2] - self.crop_size[2], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        offset_d = np.random.randint(0, margin_d + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
        crop_z1, crop_z2 = offset_d, offset_d + self.crop_size[2]

        return crop_y1, crop_y2, crop_x1, crop_x2, crop_z1, crop_z2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2, crop_z1, crop_z2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, crop_z1:crop_z2]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


# @PIPELINES.register_module()
# class RandomRotate(object):
#     """Rotate the image & seg.
#
#     Args:
#         prob (float): The rotation probability.
#         degree (float, tuple[float]): Range of degrees to select from. If
#             degree is a number instead of tuple like (min, max),
#             the range of degree will be (``-degree``, ``+degree``)
#         pad_val (float, optional): Padding value of image. Default: 0.
#         seg_pad_val (float, optional): Padding value of segmentation map.
#             Default: 255.
#         center (tuple[float], optional): Center point (w, h) of the rotation in
#             the source image. If not specified, the center of the image will be
#             used. Default: None.
#         auto_bound (bool): Whether to adjust the image size to cover the whole
#             rotated image. Default: False
#     """
#
#     def __init__(self,
#                  prob,
#                  degree,
#                  pad_val=0,
#                  seg_pad_val=255,
#                  center=None,
#                  auto_bound=False):
#         self.prob = prob
#         assert prob >= 0 and prob <= 1
#         if isinstance(degree, (float, int)):
#             assert degree > 0, f'degree {degree} should be positive'
#             self.degree = (-degree, degree)
#         else:
#             self.degree = degree
#         assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
#                                       f'tuple of (min, max)'
#         self.pal_val = pad_val
#         self.seg_pad_val = seg_pad_val
#         self.center = center
#         self.auto_bound = auto_bound
#
#     def __call__(self, results):
#         """Call function to rotate image, semantic segmentation maps.
#
#         Args:
#             results (dict): Result dict from loading pipeline.
#
#         Returns:
#             dict: Rotated results.
#         """
#
#         rotate = True if np.random.rand() < self.prob else False
#         degree = np.random.uniform(min(*self.degree), max(*self.degree))
#         if rotate:
#             # rotate image
#             results['img'] = mmcv.imrotate(
#                 results['img'],
#                 angle=degree,
#                 border_value=self.pal_val,
#                 center=self.center,
#                 auto_bound=self.auto_bound)
#
#             # rotate segs
#             for key in results.get('seg_fields', []):
#                 results[key] = mmcv.imrotate(
#                     results[key],
#                     angle=degree,
#                     border_value=self.seg_pad_val,
#                     center=self.center,
#                     auto_bound=self.auto_bound,
#                     interpolation='nearest')
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(prob={self.prob}, ' \
#                     f'degree={self.degree}, ' \
#                     f'pad_val={self.pal_val}, ' \
#                     f'seg_pad_val={self.seg_pad_val}, ' \
#                     f'center={self.center}, ' \
#                     f'auto_bound={self.auto_bound})'
#         return repr_str

# @PIPELINES.register_module()
# class SegRescale(object):
#     """Rescale semantic segmentation maps.
#
#     Args:
#         scale_factor (float): The scale factor of the final output.
#     """
#
#     def __init__(self, scale_factor=1):
#         self.scale_factor = scale_factor
#
#     def __call__(self, results):
#         """Call function to scale the semantic segmentation map.
#
#         Args:
#             results (dict): Result dict from loading pipeline.
#
#         Returns:
#             dict: Result dict with semantic segmentation map scaled.
#         """
#         for key in results.get('seg_fields', []):
#             if self.scale_factor != 1:
#                 results[key] = mmcv.imrescale(
#                     results[key], self.scale_factor, interpolation='nearest')
#         return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'
#
