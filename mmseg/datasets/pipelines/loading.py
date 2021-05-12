import os.path as osp

import mmcv
import numpy as np
import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadENVIHyperSpectralImageFromFile(object):
    """Load an ENVI Hyper Spectral Image from file.
    TODO: rewrite the helping document
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 channel_select,
                 dataset_name='cholangiocarcinoma',
                 to_float32=True,
                 normalization=True,
                 channel_to_show=(10, 20, 30),
                 median_blur=True):
        self.to_float32 = to_float32
        self.normalization = normalization
        self.dataset_name = dataset_name
        self.channel_select = channel_select
        self.channel_to_show = channel_to_show
        self.median_blur = median_blur
        self.ENVI_data_type = [None,
                               np.uint8,     # 1
                               np.int16,     # 2
                               np.int32,     # 3
                               np.float32,   # 4
                               np.float64,   # 5
                               None,
                               None,
                               None,
                               None,
                               None,
                               None,
                               np.uint16,    # 12
                               np.uint32,]   # 13


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
        hdr = dict()
        with open(filename) as f:
            for line in f.readlines():
                if '=' not in line:
                    continue
                else:
                    key, value = line.split('=')
                    key = key.strip()
                    value = value.strip()
                    hdr[key] = value
        assert hdr['file type'] == 'ENVI Standard', 'Require ENVI data: file type = ENVI Standard'
        assert hdr['byte order'] == '0', 'Require ENVI data: byte order = 0'
        assert hdr['x start'] == '0', 'Require ENVI data: x start = 0'
        assert hdr['y start'] == '0', 'Require ENVI data: y start = 0'
        assert hdr['interleave'].lower() == 'bsq', 'Require ENVI data: interleave = bsq'
        assert int(hdr['data type']) <= len(self.ENVI_data_type) and self.ENVI_data_type[int(hdr['data type'])] != None

        data_type = int(hdr['data type'])
        header_offset = int(hdr['header offset'])
        height = int(hdr['lines'])
        width = int(hdr['samples'])
        bands = int(hdr['bands'])
        if hdr['interleave'].lower() == 'bsq':
            img_bytes = np.fromfile(filename.replace('.hdr', '.raw'), dtype=self.ENVI_data_type[data_type],offset=header_offset)
            img_bytes = img_bytes.reshape((bands, height, width))
            img_bytes = img_bytes[self.channel_select, :, :]
            if self.dataset_name == 'cholangiocarcinoma':
                img_bytes = img_bytes[:,::-1,:]
            img_bytes = np.transpose(img_bytes, (1, 2, 0))
        else:
            img_bytes = np.zeros((height, width, bands), dtype=self.ENVI_data_type[data_type])
            pass
        if self.to_float32:
            img_bytes = img_bytes.astype(np.float32)
            if self.normalization:
                img_bytes -= np.mean(img_bytes,axis=(0,1),keepdims=True)
                img_bytes /= np.std(img_bytes,axis=(0,1),keepdims=True)
                ############################################3
                # img_bytes *= 16
                # img_bytes += 128
                # img_bytes = img_bytes.astype(np.uint8)
                # img_bytes = img_bytes.astype(np.float32)
                # img_bytes -= 128
                # img_bytes /= 16
                ##############################################
        if self.median_blur:
            for band in range(img_bytes.shape[0]):
                img_bytes[band, :, :] = cv2.medianBlur(img_bytes[band, :, :], ksize=3)

        results['filename'] = filename.replace('.hdr', '.png')
        results['ori_filename'] = results['img_info']['filename'].replace('.hdr', '.png')
        results['img'] = img_bytes
        results['img_shape'] = img_bytes.shape
        results['ori_shape'] = img_bytes.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img_bytes.shape
        results['scale_factor'] = 1.0
        results['channel_select'] = self.channel_select
        results['channel_to_show'] = self.channel_to_show
        num_channels = 1 if len(img_bytes.shape) < 3 else img_bytes.shape[2]
        mean = np.ones(num_channels, dtype=np.float32)*128
        std = np.ones(num_channels, dtype=np.float32)*16
        results['img_norm_cfg'] = dict(
            mean=mean,
            std=std,
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f'(dataset_name={self.dataset_name},'
        repr_str += f'(normalization={self.normalization},'
        repr_str += f'(channel_select={self.channel_select},'
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
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
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
