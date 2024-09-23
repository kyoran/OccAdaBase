import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from PIL import Image
import cv2
import time
import torch
from pyquaternion import Quaternion
from tools.optical_utils import *
from skimage.feature import hog
import sys, os, pdb

@PIPELINES.register_module()
class PreprocessOccOurs(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self, use_semantic, classes, meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
              'depth2img', 'cam2img', 'pad_shape',
              'scale_factor', 'flip', 'pcd_horizontal_flip',
              'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
              'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
              'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
              'transformation_3d_flow', 'scene_token',
              'can_bus', 'pc_range', 'occ_size', 'occ_path', 'lidar_token', 'token'
              )):
        self.use_semantic = use_semantic
        self.classes = classes
        self.meta_keys = meta_keys
        self.onehot = np.eye(len(self.classes) + 1)


    def _calc_optical(self, rgb_files):

        optical_lst = []
        factor = 6
        # raw shape: (1600, 900)
        # first_frame
        first_frame = cv2.imread(rgb_files[0])
        # mid_frame
        mid_frame = cv2.imread(rgb_files[len(rgb_files) // 2])
        # last_frame
        last_frame = cv2.imread(rgb_files[-1])

        x, y = first_frame.shape[0:2]
        first_frame = cv2.resize(first_frame, (int(y / factor), int(x / factor)))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        mid_frame = cv2.resize(mid_frame, (int(y / factor), int(x / factor)))
        mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)

        last_frame = cv2.resize(last_frame, (int(y / factor), int(x / factor)))
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        # first optical
        flows = cv2.calcOpticalFlowFarneback(
            first_frame, mid_frame, None,
            0.5, 2, 15, 2, 5, 1.2, 0)
        optical_img1 = flow_to_image(flows/255.)
        optical_lst.append(optical_img1)

        # second optical
        flows = cv2.calcOpticalFlowFarneback(
            mid_frame, last_frame, None,
            0.5, 2, 15, 2, 5, 1.2, 0)
        optical_img2 = flow_to_image(flows)  # [0, 255]
        optical_lst.append(optical_img2/255.)

        # current frame
        # print("last_frame:", last_frame.shape)
        # optical_lst.append(last_frame[..., np.newaxis])
        return optical_lst

    def _load_occ(self, path, size, gt=False):
        occ = np.load(path)
        occ = occ.astype(np.float32)

        # class 0 is 'ignore' class
        if self.use_semantic:
            occ[..., 3][occ[..., 3] == 0] = 255
        else:
            occ = occ[occ[..., 3] > 0]
            occ[..., 3] = 1
        # occ.shape: (???, 4)

        gt_occ = np.zeros(([size[0], size[1], size[2]]))
        coords = occ[:, :3].astype(np.int32)
        gt_occ[coords[:, 0], coords[:, 1], coords[:, 2]] = occ[:, 3]
        # gt_occ.shape: (200, 200, 16)
        if gt:
            return gt_occ

        gt_occ[gt_occ == 255] = 0
        final_occ = self.onehot[gt_occ.astype(np.int32)]
        # final_occ = final_occ / len(self.classes)  # it's one hot, do not need to normalize
        final_occ = final_occ.transpose(3, 0, 1, 2)
        # ForkedPdb().set_trace()

        return final_occ

    def _calc_frame_difference(self, rgb_files):
        fd_lst = []
        # factor = 10
        factor = 5
        # raw shape: (1600, 900)
        #sss = time.time()
        # first_frame
        first_frame = cv2.imread(rgb_files[0])
        # mid_frame
        mid_frame = cv2.imread(rgb_files[len(rgb_files) // 2])
        # last_frame
        last_frame = cv2.imread(rgb_files[-1])

        x, y = first_frame.shape[0:2]
        new_y = int(y / factor)
        new_x = int(x / factor)
        first_frame = cv2.resize(first_frame, (new_y, new_x))
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        mid_frame = cv2.resize(mid_frame, (new_y, new_x))
        mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)

        last_frame = cv2.resize(last_frame, (new_y, new_x))
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        # first optical
        fd_lst.append(cv2.absdiff(first_frame, mid_frame)[..., np.newaxis] / 255.)

        # second optical
        fd_lst.append(cv2.absdiff(mid_frame, last_frame)[..., np.newaxis] / 255.)

        # current frame
        # print("last_frame:", last_frame.shape)
        fd_lst.append(last_frame[..., np.newaxis] / 255.)

        # print("!!!111", fd_lst[0].shape, fd_lst[0].min(), fd_lst[0].max())
        # print("!!!222", fd_lst[1].shape, fd_lst[1].min(), fd_lst[1].max())
        # print("!!!333", fd_lst[2].shape, fd_lst[2].min(), fd_lst[2].max())
        # print("\t\tframe_difference spend:", time.time()-sss)
        #ForkedPdb().set_trace()
        return fd_lst

    def _calc_hog(self, rgb_files):
        fd_lst = []
        # factor = 10
        factor = 5
        # raw shape: (1600, 900)
        #sss = time.time()
        # first_frame
        first_frame = cv2.imread(rgb_files[0], 0)
        # mid_frame
        mid_frame = cv2.imread(rgb_files[len(rgb_files) // 2], 0)
        # last_frame
        last_frame = cv2.imread(rgb_files[-1], 0)

        x, y = first_frame.shape[0:2]
        new_y = int(y / factor)
        new_x = int(x / factor)
        first_frame = cv2.resize(first_frame, (new_y, new_x))
        # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        mid_frame = cv2.resize(mid_frame, (new_y, new_x))
        # mid_frame = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)

        last_frame = cv2.resize(last_frame, (new_y, new_x))
        # last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

        # print("last_frame.shape", last_frame.shape) # (180, 320)

        # hogs
        fd1 = hog(
            first_frame,
            orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1),
            visualize=False, feature_vector=False, channel_axis=None,
        )
        fd2 = hog(
            mid_frame,
            orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1),
            visualize=False, feature_vector=False, channel_axis=None,
        )
        fd3 = hog(
            last_frame,
            orientations=8, pixels_per_cell=(10, 10), cells_per_block=(1, 1),
            visualize=False, feature_vector=False, channel_axis=None,
        )
        # print("fd1111.shape", fd1.shape)    # (18, 32, 1, 1, 8)

        fd1 = np.transpose(fd1.squeeze(), (2, 0, 1))
        fd2 = np.transpose(fd2.squeeze(), (2, 0, 1))
        fd3 = np.transpose(fd3.squeeze(), (2, 0, 1))
        # print("fd1222.shape", fd1.shape)    # (8, 18, 32)
        # fd_lst.append(fd2-fd1)  # (h=18, w=32, 8) -> (8, 18, 32)
        # fd_lst.append(fd3-fd2)  # (h=18, w=32, 8) -> (8, 18, 32)
        fd_lst.append(fd1)
        fd_lst.append(fd2)
        fd_lst.append(fd3)
        # current frame
        # print("last_frame:", last_frame.shape)
        # fd_lst.append(last_frame[..., np.newaxis] / 255.)

        # print("!!!111", fd_lst[0].shape, fd_lst[0].min(), fd_lst[0].max())
        # print("!!!222", fd_lst[1].shape, fd_lst[1].min(), fd_lst[1].max())
        # print("!!!333", fd_lst[2].shape, fd_lst[2].min(), fd_lst[2].max())
        # print("\t\tframe_difference spend:", time.time()-sss)
        #ForkedPdb().set_trace()
        return fd_lst


    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """

        meta_keys = {}
        for key in self.meta_keys:
            if key in results:
                meta_keys[key] = results[key]

        # 1. occ
        # occ_path = [
        #     T_3_info['occ_path'],
        #     T_2_info['occ_path'],
        #     T_1_info['occ_path'],
        #     t_1_info['occ_path'],
        #     T_info['occ_path'],
        #     t1_info['occ_path'],
        #     t2_info['occ_path'],
        #     t3_info['occ_path'],
        #     t4_info['occ_path'],
        #     t5_info['occ_path'],
        data = {}

        # 1. inter_motion_frames
        new_IFs = []
        for one_IF in results['IFs']:
            motion_frames = {}
            fff = 6
            if not one_IF:
                for cam_type in ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                    motion_frames[cam_type] = [torch.zeros([900//fff, 1600//fff, 3], dtype=torch.float) for _ in range(2)]
                    # motion_frames[cam_type] = [torch.zeros([8, 18, 32], dtype=torch.float) for _ in range(3)]
                    # print("ERROR, not have new_IFs1")

            else:
                for cam_type, cam_file in one_IF.items():
                    #print("\tnow at:", cam_type, len(cam_file))
                    # ForkedPdb().set_trace()
                    # import pdb; pdb.set_trace()
                    # s = time.time()
                    if len(cam_file) == 0:
                        one_cam_motion_frame = [torch.zeros([900//fff, 1600//fff, 3], dtype=torch.float) for _ in range(2)]
                        # motion_frames[cam_type] = [torch.zeros([8, 18, 32], dtype=torch.float) for _ in range(3)]
                        # print("!!!!!!!!!!!!!!!!!!!!!!!CAM_FILE=0")
                        # print("ERROR, not have new_IFs2")

                    else:
                        one_cam_motion_frame = self._calc_optical(cam_file)
                        # one_cam_motion_frame = self._calc_frame_difference(cam_file)
                        # one_cam_motion_frame = self._calc_hog(cam_file)

                    # print("one cam's optical spend:", time.time()-s)
                    motion_frames[cam_type] = one_cam_motion_frame
            new_IFs.append(motion_frames)


        t_occ = []
        t_occ_gt = []
        for i in range(len(results['occ_path'])):
            t_occ.append(
                self._load_occ(results['occ_path'][i], results['occ_size'])
            )
            t_occ_gt.append(
                self._load_occ(results['occ_path'][i], results['occ_size'], gt=True)    # (17, 200, 200, 16)
            )

        dt_trans = []
        dt_rotat = []
        for i in range(len(results['translation'])-1):
            
            prev_trans = np.array(results['translation'][i])
            next_trans = np.array(results['translation'][i+1])
            dt_trans.append(next_trans - prev_trans)


            prev_rotat = np.array(results['rotation'][i])
            prev_rotat = np.array(list(Quaternion(prev_rotat).yaw_pitch_roll))
            next_rotat = np.array(results['rotation'][i+1])
            next_rotat = np.array(list(Quaternion(next_rotat).yaw_pitch_roll))
            dt_rotat.append(next_rotat - prev_rotat)
                

        # return
        data['IFs'] = new_IFs
        data['t_occ'] = t_occ
        data['t_occ_gt'] = t_occ_gt
        data['dt_trans'] = dt_trans
        data['dt_rotat'] = dt_rotat
        data['info_length'] = results['info_length']

        data['meta_keys'] = DC(meta_keys, cpu_only=True)
        #print("\t\tpreprocess done")
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + f'(keys={self.keys})'