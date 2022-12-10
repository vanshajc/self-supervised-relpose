# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

from scipy.spatial.transform import Rotation as R

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

        train_folders = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

        curr_folder = train_folders 
        self.all_gt_poses = {}

        for sequence in curr_folder:
            print("Loading ", sequence)
            # all_gt_poses[int(sequence)] = {0: np.array([0, 0, 0, 0, 0, 0, 1.0])}
            with open(os.path.join(self.data_path, f'poses/{sequence}.txt'), 'r') as f:
                gt_poses_path = os.path.join(self.data_path, "poses", f"{sequence}.txt")
                gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
                gt_global_poses = np.concatenate(
                    (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
                gt_global_poses[:, 3, 3] = 1

                self.all_gt_poses[int(sequence)] = gt_global_poses


    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path

    def get_pose(self, folder, frame_index, other_frame_index, side):
        try:
            global_pose = self.all_gt_poses[int(folder)][frame_index]
            global_pose2 = self.all_gt_poses[int(folder)][other_frame_index]
        except:
            import pdb; pdb.set_trace()

        if other_frame_index < frame_index:
            tmp_p = global_pose.copy()
            global_pose = global_pose2.copy()
            global_pose2 = tmp_p

        p = np.linalg.inv(np.dot(np.linalg.inv(global_pose2), global_pose))
        p_xyz = p[:3, 3]
        p_R = p[:3, :3]
        p_q = R.from_matrix(p_R).as_quat()
        rel_pose = np.concatenate([p_xyz, p_q])
        return rel_pose

    def get_corresp(self, folder, frame_index, frame_id, side):
        f_str = "{:06d}{}".format(frame_index, ".npz")
        corresp_path = os.path.join(
            self.data_path,
            "superglue_poses",
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        corresp = np.load(corresp_path, allow_pickle=True)
        corresp_dict = corresp['arr_0'].item()
        rot = corresp_dict[("superglue_rot", frame_id)]
        trans = corresp_dict[("superglue_trans", frame_id)]
        valid = corresp_dict[("superglue_valid", frame_id)]
        if frame_id < 0:
            rot = R.from_quat(rot).inv().as_quat()
            trans = -trans
        rel_pose = np.concatenate([trans, rot])
        return rel_pose, valid


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
