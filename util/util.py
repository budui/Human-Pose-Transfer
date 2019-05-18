import torch
from PIL import Image
import numpy as np

from util.pose import draw_pose_from_cords_and_visibility, draw_pose_from_cords


def tensor2image(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def get_current_visuals(img_path, data_pair, new_imgs=None):
    height, width, batch_size = data_pair["P1"].size(2), data_pair["P1"].size(3), data_pair["P1"].size(0)

    image_num = batch_size if batch_size < 6 else 6

    vis = np.zeros((height * image_num, width * (5 + len(new_imgs)), 3)).astype(np.uint8)

    def make_vis(image_list, row_id):
        for img_id, img in enumerate(image_list):
            vis[height * row_id:height * (1 + row_id), width * img_id:width * (img_id + 1), :] = img

    for i in range(image_num):
        new_img_list = []
        if new_imgs is not None:
            for nimg in new_imgs:
                nimg.clamp_(-1, 1)
                new_img_list.append(tensor2image(nimg.data[i]))
        input_p1 = tensor2image(data_pair["P1"].data[i])
        image_size = input_p1.shape[:2]
        input_p2 = tensor2image(data_pair["P2"].data[i])
        input_p2_mask = tensor2image(data_pair["MP2"].data[i])
        input_bp1 = draw_pose_from_cords(data_pair["KP1"].data[i], image_size)[0]
        input_bp2 = draw_pose_from_cords(data_pair["KP2"].data[i], image_size)[0]

        make_vis([input_p1, input_bp1, input_p2, input_bp2, input_p2_mask] + new_img_list, i)

    save_image(vis, img_path)


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
    save_image(vis, save_path)