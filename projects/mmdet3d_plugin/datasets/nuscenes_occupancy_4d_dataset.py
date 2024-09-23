import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
import pdb, os
from nuscenes.nuscenes import NuScenes  # add


@DATASETS.register_module()
class CustomNuScenesOcc4dDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_test = overlap_test
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self._set_group_flag()
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=False)  # add
        
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):  # modify by Yufc, add 4d information
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        # input_dict = dict(
        #     occ_path=info['occ_path'], # 这里就只获取了一个，需要获取多个
        #     occ_size = np.array(self.occ_size),
        #     pc_range = np.array(self.pc_range)
        # )
        
        # do sth here to add 4d information
        input_dict = dict(
            token=info['token'],
            scene_token=info['scene_token'],
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range)
        )
        T0_info = self.data_infos[index]  # 当前时刻的 info
        T1_info = self.data_infos[index - 1] if index - 1 > 0 else None
        T2_info = self.data_infos[index - 2] if index - 2 > 0 else None
        T3_info = self.data_infos[index - 3] if index - 3 > 0 else None
        
        # 处理越界(跑到别人的场景下了)
        if T1_info is not None and T1_info['scene_token'] != T0_info['scene_token']:
            T1_info, T2_info, T3_info = None, None, None  # 跑到别人那一个场景下了
        if T2_info is not None and T2_info['scene_token'] != T0_info['scene_token']:
            T2_info, T3_info = None, None
        if T3_info is not None and T3_info['scene_token'] != T0_info['scene_token']:
            T3_info = None
        
        # 什么时候可以训练? T0-T3都不是None的时候可以训练
        if T1_info == None or T2_info == None or T3_info == None or T0_info == None:
            return None
        
        occ_path_lst = [
            T0_info['occ_path'],
            T1_info['occ_path'],
            # T2_info['occ_path'],
            # T3_info['occ_path']
        ]  # 最后要返回的 occ_path 列表, 需要注意顺序!
        
        prev_tracking_points = [T1_info]
        tracking_points = [T1_info, T0_info]
        
        # 拿到中间帧
        # Track
        dir = 'next'
        IFs = []
        if self.modality['use_camera']:
            # tracking_points = [T1_info]  # 如果有还有过去的，就是 [..., T2_info, T1_info]
            # if type == "test":
            #     tracking_points = [t_1_info, T_info, t1_info, t2_info, t3_info, t4_info, t5_info, t6_info]
            for i in range(0, len(tracking_points)):
                one_start_info = tracking_points[i]
                if(i == len(tracking_points) - 1):
                    break  # [T1_info, T0_info] 只需要获取除了最后一个之外的其他 tracking_points 的后面中间帧
                intermediate_filenames = {}
                for cam_type, cam_info in one_start_info['cams'].items():
                    intermediate_filenames[cam_type] = []
                    one_cam_sample_data_token = cam_info['sample_data_token']
                    one_sample_cam = self.nusc.get('sample_data', one_cam_sample_data_token)
                    assert one_sample_cam["is_key_frame"] is True
                    non_key_sample = self.nusc.get('sample_data', one_sample_cam[dir])    
                    mmm = 1
                    while not non_key_sample["is_key_frame"]:
                        if non_key_sample[dir] == '':
                            break
                        mmm += 1
                        non_key_sample = self.nusc.get('sample_data', non_key_sample[dir])
                        # if cam_type == "CAM_BACK":
                        #     print("\t\tnon_key_sample:", non_key_sample)

                        intermediate_filenames[cam_type].append(
                            os.path.join(self.data_root, non_key_sample["filename"])
                        )      
                IFs.append(intermediate_filenames)      
        
        # 拿到 now 时刻的图片信息
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix

                if 'lidar2cam' in cam_info.keys():
                    lidar2cam_rt = cam_info['lidar2cam'].T
                else:
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3,:3] = lidar2cam_r.T
                    lidar2cam_rt[3,:3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0],:intrinsic.shape[1]] = intrinsic

                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))
            # 处理 prev 的
            prev_image_paths = []
            prev_lidar2img_rts = []
            prev_lidar2cam_rts = []
            prev_cam_intrinsics = []
            for i in range(0, len(prev_tracking_points)):
                cur_info = prev_tracking_points[i]
                image_paths = []
                lidar2img_rts = []
                lidar2cam_rts = []
                cam_intrinsics = []
                for cam_type, cam_info in cur_info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix

                    if 'lidar2cam' in cam_info.keys():
                        lidar2cam_rt = cam_info['lidar2cam'].T
                    else:
                        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                        lidar2cam_t = cam_info[
                            'sensor2lidar_translation'] @ lidar2cam_r.T
                        lidar2cam_rt = np.eye(4)
                        lidar2cam_rt[:3,:3] = lidar2cam_r.T
                        lidar2cam_rt[3,:3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0],:intrinsic.shape[1]] = intrinsic

                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                    cam_intrinsics.append(viewpad)
                    lidar2cam_rts.append(lidar2cam_rt.T)

                prev_image_paths.append(image_paths)
                prev_lidar2img_rts.append(lidar2img_rts)
                prev_lidar2cam_rts.append(lidar2cam_rts)
                prev_cam_intrinsics.append(cam_intrinsics)
                
            input_dict.update(
                dict(
                    img_filename_prev=prev_image_paths,
                    lidar2img_prev=prev_lidar2img_rts,
                    cam_intrinsic_prev=prev_lidar2cam_rts,
                    lidar2cam_prev=prev_cam_intrinsics,
                )) 

        input_dict.update(
            dict(
                occ_path=occ_path_lst,
                IFs=IFs
            )
        )

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            info = self.data_infos[idx]
            while True:
                # 这里的data可能是空，要处理 #BUG
                data = self.prepare_test_data(idx)
                if data is not None: 
                    return data
                else:
                    idx = self._rand_another(idx)
                    continue
        while True:
            data = self.prepare_train_data(idx)  # 这个就是example给过来的
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {0: 'IoU'}
            class_num = len(self.class_names) + 1
            for i, name in enumerate(self.class_names):
                class_names[i + 1] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0)
            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])

        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict = {'Acc':results[0],
                          'Comp':results[1],
                          'CD':results[2],
                          'Prec':results[3],
                          'Recall':results[4],
                          'F-score':results[5]}

        return results_dict
