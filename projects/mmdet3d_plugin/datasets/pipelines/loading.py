# import open3d as o3d
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os


@PIPELINES.register_module()
class MultiLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # 这里的逻辑是 now 时刻的
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
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
        # 现在处理 prev 的时刻
        prev_file_name_list = []
        prev_img_list = []
        prev_img_shape_list = []
        prev_ori_shape_list = []
        prev_pad_shape_list = []
        prev_img_norm_cfg_list = []
        for i in range(0, len(results['img_filename_prev'])):
            filename = results['img_filename_prev'][i]
            # img is of shape (h, w, c, num_views)
            img = np.stack(
                [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
            if self.to_float32:
                img = img.astype(np.float32)
            # results['filename'] = filename
            prev_file_name_list.append(filename)
            
            # unravel to list, see `DefaultFormatBundle` in formating.py
            # which will transpose each image separately and then stack into array
            # results['img'] = [img[..., i] for i in range(img.shape[-1])]
            prev_img_list.append([img[..., i] for i in range(img.shape[-1])])
            # results['img_shape'] = img.shape
            prev_img_shape_list.append(img.shape)
            # results['ori_shape'] = img.shape
            prev_ori_shape_list.append(img.shape)
            
            # Set initial values for default meta_keys
            # results['pad_shape'] = img.shape
            prev_pad_shape_list.append(img.shape)
            results['scale_factor'] = 1.0  # 这个不需要特殊处理
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            # results['img_norm_cfg'] = dict(
            #     mean=np.zeros(num_channels, dtype=np.float32),
            #     std=np.ones(num_channels, dtype=np.float32),
            #     to_rgb=False)
            prev_img_norm_cfg_list.append(dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
            # 现在处理 prev 的时刻
        results['filename_prev'] = prev_file_name_list
        results['img_prev'] = prev_img_list
        results['img_shape_prev'] = prev_img_shape_list
        results['ori_shape_prev'] = prev_ori_shape_list
        results['pad_shape_prev'] = prev_pad_shape_list
        results['img_norm_cfg_prev'] = prev_img_norm_cfg_list
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class. 
    """

    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic
    
    def __call__(self, results):  # modify by Yufc
        occ_path_lst = results['occ_path']
        assert(len(occ_path_lst) == len(results['img_prev']) + 1)
        
        occ_now_path = occ_path_lst[0]  # 第一个是现在的
        occ_prev_path_lst = occ_path_lst[-len(results['img_prev']):] # 这些是之前的
        assert(len(occ_prev_path_lst) == len(results['img_prev']))

        # now 时刻的 occ
        occ_now = np.load(occ_now_path)
        occ_now = occ_now.astype(np.float32)
        # class 0 is 'ignore' class
        if self.use_semantic:
            occ_now[..., 3][occ_now[..., 3] == 0] = 255
        else:
            occ_now = occ_now[occ_now[..., 3] > 0]
            occ_now[..., 3] = 1
        
        results['gt_occ'] = occ_now
        
        # 处理prev时刻的occ
        occ_prev_lst = []
        for i in range(0, len(results['img_prev'])):
            occ_prev_time = np.load(occ_prev_path_lst[i])
            occ_prev_time = occ_prev_time.astype(np.float32)
            # class 0 is 'ignore' class
            if self.use_semantic:
                occ_prev_time[..., 3][occ_prev_time[..., 3] == 0] = 255
            else:
                occ_prev_time = occ_prev_time[occ_prev_time[..., 3] > 0]
                occ_prev_time[..., 3] = 1
            occ_prev_lst.append(occ_prev_time)
        
        results['gt_occ_prev'] = occ_prev_lst

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

