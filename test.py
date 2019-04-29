#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser

import util.util as util
import dataset.dataset as dataset

import models.PG2 as PG2


def get_current_visuals(img_folder, data_pair, generated_img1, generated_img2, idx):
    height, width, batch_size = data_pair["P1"].size(2), data_pair["P1"].size(3), data_pair["P1"].size(0)

    vis = np.zeros((height * batch_size, width * 6, 3)).astype(np.uint8)  # h, w, c

    for i in range(batch_size):
        input_P1 = util.tensor2im_(data_pair["P1"].data[i])
        input_P2 = util.tensor2im_(data_pair["P2"].data[i])
        fake_p1 = util.tensor2im_(generated_img1.data[i])
        fake_p2 = util.tensor2im_(generated_img2.data[i])
        input_BP1 = util.draw_pose_from_map_(data_pair["BP1"].data[i])[0]
        input_BP2 = util.draw_pose_from_map_(data_pair["BP2"].data[i])[0]

        vis[height*i:height*(i+1), :width, :] = input_P1
        vis[height*i:height*(i+1):, width:width * 2, :] = input_BP1
        vis[height*i:height*(i+1):, width * 2:width * 3, :] = input_P2
        vis[height*i:height*(i+1):, width * 3:width * 4, :] = input_BP2
        vis[height*i:height*(i+1):, width * 4:width * 5, :] = fake_p1
        vis[height*i:height*(i+1):, width * 5:, :] = fake_p2

    img_path = os.path.join(img_folder, "batch_{}.png".format(idx))
    print("save train image: {}".format(img_path))
    util.save_image(vis, img_path)


def generate(img_folder, g1_model, g2_model, data_pair, idx):
    condition_img = data_pair["P1"]
    new_pose = data_pair["BP2"]

    g1_img = g1_model(torch.cat([condition_img, new_pose], dim=1))
    diff_img = g2_model(torch.cat([condition_img, g1_img], dim=1))
    g2_img = g1_img + diff_img

    get_current_visuals(img_folder, data_pair, g1_img, g2_img, idx)


def move_data_pair_to(device, data_pair):
    # move data to GPU
    for k in data_pair:
        if "path" in k:
            # do not move string
            continue
        else:
            data_pair[k] = data_pair[k].to(device)


def main():
    parser = ArgumentParser(description='Training')
    parser.add_argument('--gpu_id', default=2, type=int, help='gpu_id: e.g. 0')
    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    cudnn.benchmark = True
    device = 'cuda'

    image_dataset = dataset.BoneDataset(
        "../DataSet/Market-1501-v15.09.15/bounding_box_test/",
        "data/market/test/pose_map_image/",
        "data/market/test/pose_mask_image/",
        "data/market-pairs-test.csv",
        random_select=True,
        random_select_size=100
    )

    #  BATCH_SIZE MUST TO SET AS 1
    image_loader = DataLoader(image_dataset, batch_size=4, num_workers=1)
    img_folder = "data/market/generated_img/"

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    g1_model = PG2.G1(3 + 18, repeat_num=5, half_width=True, middle_z_dim=64)
    g1_model.load_state_dict(torch.load("checkpoints/G1.pth"))
    g1_model = g1_model.to(device)

    g2_model = PG2.G2(3 + 3, hidden_num=64, repeat_num=3, skip_connect=1)
    g2_model.load_state_dict(torch.load("checkpoints/G2.pth"))
    g2_model = g2_model.to(device)

    for idx, data_pair in enumerate(image_loader):
        move_data_pair_to(device, data_pair)
        generate(img_folder, g1_model, g2_model, data_pair, idx)

if __name__ == '__main__':
    main()