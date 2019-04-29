import torch
import numpy as np

import util.util as util
from skimage.draw import circle, line_aa

# draw pose img
LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1

IMAGE_SIZE = (128, 64)


def show(poses, save_path, pose_has_norm=True):
    vis = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1] * len(poses), 3)).astype(np.uint8)
    for i, p in enumerate(poses):
        pose_joints, _ = p.split([2, 1], dim=-1)
        if pose_has_norm:
            pose_joints = pose_joints.mul(torch.Tensor([IMAGE_SIZE]).expand([18, 2]))
        colors = draw_pose_from_cords(pose_joints, IMAGE_SIZE)
        vis[:,IMAGE_SIZE[1]*i:IMAGE_SIZE[1]*(i+1),:] = colors
    util.save_image(vis, save_path)


# draw pose from map
def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]

    return colors

