import numpy as np
import torch
from skimage.draw import circle, line_aa

import util.util as util

# draw pose img
# ref https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_18.png
LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
            [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17], [2, 16], [5, 17]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
          'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

# 7 Body Regions-Of-Interest from <Spindle Net: Person Re-identification with Human Body Region
# Guided Feature Decomposition and Fusion>
# seven body sub-regions, including three macro sub-regions (head-shoulder,
# upper body, lower body) and four micro sub-regions (two arms, two legs)
BODY_ROI_7 = [
    [0, 1, 2, 5, 14, 15, 16, 17],
    [2, 3, 4, 5, 6, 7, 8, 11],
    [8, 9, 10, 11, 12, 13],
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10],
    [11, 12, 13]
]


MISSING_VALUE = -1


def _is_invalid(point, visibility, mode="gen", image_size=(128, 64)):
    if mode == "origin":
        return MISSING_VALUE in point
    elif mode == "gen":
        if visibility > 0.5 and (0 <= point[0] < image_size[0] and 0 <= point[1] < image_size[1]):
            return False
        return True
    else:
        raise NotImplementedError("invalid mode <{}> for _check_valid".format(mode))


def show_with_visibility(poses, save_path, pose_has_norm=True, image_size=(128, 64)):
    # not a batch of pose
    if len(poses[0].size()) == 2:
        vis = np.zeros((image_size[0], image_size[1] * len(poses), 3)).astype(np.uint8)
        batch_pose = False
    elif len(poses[0].size()) == 3:
        vis = np.zeros((image_size[0] * poses[0].size(0), image_size[1] * len(poses), 3)).astype(np.uint8)
        batch_pose = True
    else:
        raise TypeError("expect `poses` is a list of Tensor of 18*3 "
                        "or batch_size*18*3, but get {}".format(poses[0].size()))

    for i, p in enumerate(poses):
        p = p.to("cpu")
        pose_joints, visibility = p.split([2, 1], dim=-1)
        if pose_has_norm:
            pose_joints = pose_joints.mul(torch.Tensor([image_size]).expand([18, 2]))

        pose_joints = pose_joints.to(torch.int)
        if batch_pose:
            for bi in range(len(pose_joints)):
                per_pose_in_batch = pose_joints[bi]
                per_v = visibility[bi]
                colors = draw_pose_from_cords_and_visibility(per_pose_in_batch, per_v, image_size)
                vis[bi * image_size[0]:(bi + 1) * image_size[0], image_size[1] * i:image_size[1] * (i + 1), :] = colors
        else:
            colors = draw_pose_from_cords_and_visibility(pose_joints, visibility, image_size)
            vis[:, image_size[1] * i:image_size[1] * (i + 1), :] = colors
    util.save_image(vis, save_path)


# draw pose from map
def draw_pose_from_cords_and_visibility(pose_joints, visibility, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3,), dtype=np.uint8)

    if draw_joints:
        for f, t in LIMB_SEQ:
            if _is_invalid(pose_joints[f], visibility[f]) or _is_invalid(pose_joints[t], visibility[t]):
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            try:
                colors[yy, xx] = np.expand_dims(val, 1) * 255
            except IndexError:
                print(pose_joints[f], pose_joints[t])

    for i, joint in enumerate(pose_joints):
        if _is_invalid(pose_joints[i], visibility[i]):
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]

    return colors


def _test():
    pose = torch.ones([18, 3])
    pose = pose * 0.5
    show_with_visibility([pose], "./pose.jpg")


if __name__ == '__main__':
    _test()
