import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rgb_to_grayscale
from tqdm import tqdm

from datasets import KITTIOdomDataset
from superglue.matching import Matching as SuperGlueMatching
from superglue.utils import estimate_pose
from utils import readlines


if __name__ == "__main__":
    frame_ids = [0, 1, -1]
    scales = [0]
    height = 192
    width = 640
    datapath = "../data/kitti_odom"
    output_dir = os.path.join(datapath, "superglue_poses")

    fpath = os.path.join("splits", "odom", "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    val_filenames = readlines(fpath.format("val"))
    img_ext = '.jpg'

    train_dataset = KITTIOdomDataset(
            datapath, train_filenames, height, width,
            frame_ids, len(scales), is_train=True, img_ext=img_ext)
    train_loader = DataLoader(
            train_dataset, 1, True,
            num_workers=12, pin_memory=True, drop_last=False
    )

    matcher = SuperGlueMatching().cuda()
    for batch_idx, inputs in tqdm(enumerate(train_loader)):
        line = inputs["filename"][0].split()
        side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        
        f_str = "{:06d}.npz".format(frame_index)
        seq_output_dir = os.path.join(
            output_dir,
            "sequences",
            "{:02d}".format(int(folder)),
            "image_{}".format(side_map[side])
        )
        if not os.path.exists(seq_output_dir):
            os.makedirs(seq_output_dir)
        output_file = os.path.join(seq_output_dir, f_str)

        pose_feats = {f_i: inputs["color", f_i, 0] for f_i in frame_ids}
        K1 = K2 = inputs[('K', 0)].numpy()
        
        # batch size is 1
        output_dict = {}
        for frame_id in frame_ids[1:]:
            superglue_data = {
                'image0': rgb_to_grayscale(inputs[('color', 0, 0)][0, None]).cuda(),
                'image1': rgb_to_grayscale(inputs[('color', frame_id, 0)][0, None]).cuda()
            }
            preds = matcher(superglue_data)
            matches = preds['matches0']
            valid = matches > -1
            mkpts0 = preds['keypoints0'][0][valid[0]].detach().cpu().numpy()
            mkpts1 = preds['keypoints1'][0][matches[0, valid[0]]].detach().cpu().numpy()
            ret = estimate_pose(mkpts0, mkpts1, K1[0], K2[0], 1)

            # can't find an R, t
            if ret is None:
                rel_rot_quat = np.zeros(4)
                rel_trans = np.zeros(3)
            else:
                rel_rot, rel_trans = ret[0], ret[1]
                rel_rot_quat = R.from_matrix(rel_rot).as_quat()

            output_dict[("superglue_rot", frame_id)] = rel_rot_quat
            output_dict[("superglue_trans", frame_id)] = rel_trans
            output_dict[("superglue_valid", frame_id)] = ret is not None
        np.savez(output_file, output_dict)