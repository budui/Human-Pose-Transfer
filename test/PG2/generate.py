import os

import torch
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events

from dataset.bone_dataset import BoneDataset
from models import PG2
import util.util as util

from test.helper import move_data_pair_to


def _pg2_generate(G1_path, G2_path, device):
    generator_1 = PG2.G1(3 + 18, repeat_num=5, half_width=True, middle_z_dim=64)
    generator_1.load_state_dict(torch.load(G1_path))
    generator_2 = PG2.G2(3 + 3, hidden_num=64, repeat_num=3, skip_connect=1)
    generator_2.load_state_dict(torch.load(G2_path))
    generator_1.to(device)
    generator_2.to(device)

    def generator(batch):
        move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        condition_pose = batch["BP2"]

        # get generated img
        generator_1_imgs = generator_1(torch.cat([condition_img, condition_pose], dim=1))
        diff_imgs = generator_2(torch.cat([condition_img, generator_1_imgs], dim=1))
        generated_imgs = generator_1_imgs + diff_imgs
        return generated_imgs

    return generator


def get_tester(option, device):
    output_dir = option.output_dir
    generate = _pg2_generate(option.G1_path, option.G2_path, device)

    def step(engine, batch):
        generated_imgs = generate(batch)
        condition_names = batch["P1_path"]
        target_names = batch["P2_path"]

        for i in range(generated_imgs.size(0)):
            img_name = "{}#{}.jpg".format(condition_names[i][:-4], target_names[i][:-4])
            img = util.tensor2im_(generated_imgs.data[i])
            util.save_image(img, os.path.join(output_dir, img_name))
        return

    tester = Engine(step)

    pbar = ProgressBar()
    pbar.attach(tester)

    @tester.on(Events.STARTED)
    def mkdir(engine):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    return tester


def add_new_arg_for_parser(parser):
    parser.add_argument("--pair_path", type=str, default="data/market-pairs-test.csv")
    parser.add_argument('--market1501', type=str, default="../dataset/Market-1501-v15.09.15/")
    parser.add_argument("--G1_path", type=str)
    parser.add_argument("--G2_path", type=str)


def get_data_loader(opt):
    image_dataset = BoneDataset(
        os.path.join(opt.market1501, "bounding_box_test/"),
        "data/market/test/pose_map_image/",
        "data/market/test/pose_mask_image/",
        opt.pair_path,
        random_select=False
    )
    print("load test dataset: {} pairs".format(len(image_dataset)))
    image_loader = DataLoader(image_dataset, batch_size=opt.batch_size, num_workers=8)
    return image_loader