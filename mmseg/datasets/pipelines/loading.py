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

                # img_bytes -= np.mean(img_bytes,axis=(0,1),keepdims=True)
                # img_bytes /= np.clip(np.std(img_bytes,axis=(0,1),keepdims=True), 1e-6, 1e6)
                ############################################3
                img_bytes -= np.min(img_bytes)
                img_bytes /= np.max(img_bytes)
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
class LoadENVIHyperSpectralImageFromFile_DKJSB(object):
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

        # self.mean = [8767.74507753, 8704.33528893, 8764.76146787, 8823.74236313, 8859.69514695,
        #              8837.16513246, 8792.0379876 , 8704.95183511, 8533.700792  , 8480.12577875,
        #              8407.32103156, 8318.1674773 , 8290.49329004, 8251.34573434, 8240.06533973,
        #              8199.50589202, 8093.38762291, 8112.23262798, 8129.81115241, 8132.94929126,
        #              8230.86552891, 8354.09237416, 8504.05907384, 8662.11308906, 8818.7432458,
        #              8983.27861134, 9155.95266069, 9292.61352853, 9422.36312335, 9555.92253692,
        #              9640.53238117, 9694.76821996, 9795.0561307 , 9819.82870302, 9847.22004009,
        #              9876.4136438 , 9905.60690625, 9896.89131326, 9926.73061522, 9924.84012027,
        #              9953.75414584, 9919.14214838, 9855.66764303, 9853.46132631, 9827.25613361,
        #              9885.0022777 , 9833.54688764, 9820.1838158 , 9828.67573733, 9787.00645107,
        #              9791.73297982, 9748.44445561, 9737.40782204, 9765.91671089, 9764.43367338,
        #              9756.60531767, 9762.92881418, 9758.20744925, 9760.1832485 , 9761.23269676],
        # self.std =  [ 786.40690148,  646.46209235,  644.47730128,  646.71471702,  665.09259162,
        #               709.65204137,  788.234844  ,  880.64193853,  975.27485053, 1068.31559186,
        #              1137.6271906 , 1192.3429918 , 1226.22350158, 1255.65266396, 1307.78852952,
        #              1364.20851072, 1429.02842811, 1496.99072172, 1540.5701182 , 1562.97189646,
        #              1559.87362553, 1531.76255772, 1488.65268418, 1448.0766883 , 1405.53092211,
        #              1361.08308119, 1321.32763201, 1275.8323443 , 1222.28518207, 1164.17655183,
        #              1089.87472044, 1037.55694709,  989.79810197,  944.67915937,  916.67531173,
        #               893.8713549 ,  877.89463819,  855.60262008,  842.5042363 ,  817.46170164,
        #               778.36017188,  718.42901102,  652.82189621,  596.81521143,  545.53864585,
        #               516.38844564,  490.72142761,  468.79087104,  446.6625361 ,  426.24756705,
        #               414.9572136 ,  390.64571949,  376.65323326,  360.64912455,  349.05100564,
        #               342.1106682 ,  338.48868696,  336.2702614 ,  334.04175279,  331.07240439],

        self.mean = [8920.18366902, 8652.57078643, 8705.88716656, 8761.31643149, 8797.92747553,
                     8756.64686044, 8681.5015259 , 8603.40946583, 8429.26601425, 8343.55937431,
                     8237.95588917, 8161.94817246, 8141.1642471 , 8092.35520184, 8058.31556385,
                     8029.59344101, 7915.33230388, 7934.56899851, 7913.5090089 , 7933.60020121,
                     8045.92852674, 8171.39256948, 8327.8225149 , 8496.60679838, 8661.47454707,
                     8840.49101199, 9006.41110208, 9155.68866982, 9296.41343159, 9444.32198905,
                     9522.2987977 , 9616.03092596, 9690.21682943, 9704.51227563, 9753.22937012,
                     9780.3339864 , 9813.2747978 , 9811.20689707, 9835.20388567, 9841.87270785,
                     9862.6220361 , 9794.56231856, 9838.84450898, 9804.18840299, 9779.03623766,
                     9811.65460938, 9766.08784571, 9767.40317176, 9760.93315795, 9728.11994834,
                     9737.63436383, 9711.85823024, 9701.88658967, 9727.10462146, 9719.39775326,
                     9722.32486557, 9726.10685794, 9722.31715204, 9722.80703719, 9724.01522073]
        self.std =  [2562.14614101,  832.97011692,  829.78701364,  828.7445067 ,  852.52808174,
                     901.25077171 ,1010.34699124 ,1127.80544775 ,1231.67180892 ,1356.85291226,
                     1409.99260904, 1472.41382974, 1508.30225424, 1531.62800864, 1601.91933828,
                     1679.8181784 , 1750.34595475, 1852.29546423, 1862.61630224, 1891.73400702,
                     1869.09546926, 1824.87407107, 1771.21978474, 1724.41482772, 1685.41510251,
                     1635.44493289, 1589.14723024, 1543.6997826 , 1472.99119966, 1408.28806211,
                     1351.30544249, 1345.69031082, 1274.20578121, 1233.02056739, 1212.67449583,
                     1196.37016345, 1193.11247864, 1169.27930436, 1167.4473712 , 1127.35309462,
                     1072.89175586, 1017.56080766,  980.44655572,  867.0781352 ,  818.96612374,
                     758.53384993 , 716.99598273 , 707.12476691 , 671.80858088 , 648.36837253,
                     637.90684873 , 604.50683842 , 594.7168734  , 571.17602115 , 553.53120222,
                     546.9324421  , 541.89280019 , 540.19842938 , 538.25427469 , 534.81303441]

        self.mean = np.array(self.mean, dtype=np.float32).reshape((1, 1, -1))
        self.std = np.array(self.std, dtype=np.float32).reshape((1, 1, -1))

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

                img_bytes -= self.mean[..., self.channel_select]
                img_bytes /= self.std[..., self.channel_select]
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
