import os

import numpy as np
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader

import util.util as util
from dataset.bone_dataset import BoneDataset


def select_generator(option, device):
    if option.name == "PG2-Generate":
        from test.PG2 import get_generator
        return get_generator(option.G1_path, option.G2_path, device)
    else:
        raise NotImplementedError("not implemented generate methods: {}".format(option.name))


def get_tester(option, device):
    output_dir = option.output_dir
    generate = select_generator(option, device)

    def step(engine, batch):
        generated_imgs = generate(batch)
        condition_names = batch["P1_path"]
        target_names = batch["P2_path"]

        for i in range(generated_imgs.size(0)):
            # image height and width
            image_size = (generated_imgs.size(2), generated_imgs.size(3))
            image = np.zeros((image_size[0], image_size[1] * 3, 3)).astype(np.uint8)
            image[:, 0 * image_size[1]:1 * image_size[1], :] = util.tensor2image(batch["P1"].data[i])
            image[:, 1 * image_size[1]:2 * image_size[1], :] = util.tensor2image(batch["P2"].data[i])
            image[:, 2 * image_size[1]:3 * image_size[1], :] = util.tensor2image(generated_imgs.data[i])
            util.save_image(
                image,
                os.path.join(output_dir, "{}___{}_vis.jpg".format(condition_names[i], target_names[i]))
            )
        return

    tester = Engine(step)

    pbar = ProgressBar()
    pbar.attach(tester)

    @tester.on(Events.STARTED)
    def mkdir(engine):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    return tester


def add_new_arg_for_parser(parser, name):
    parser.add_argument("--pair_path", type=str, default="data/market/pairs-test.csv")
    parser.add_argument('--market1501', type=str, default="../dataset/Market-1501-v15.09.15/")
    if name == "PG2-Generate":
        parser.add_argument("--G1_path", type=str, default="./data/market/models/PG2/G1.pth")
        parser.add_argument("--G2_path", type=str)


def get_data_loader(opt):
    image_dataset = BoneDataset(
        os.path.join(opt.market1501, "bounding_box_test/"),
        "data/market/test/pose_map_image/",
        "data/market/test/pose_mask_image/",
        opt.pair_path,
        "data/market/annotation-test.csv",
    )
    print("load test dataset: {} pairs".format(len(image_dataset)))
    image_loader = DataLoader(image_dataset, batch_size=opt.batch_size, num_workers=8)
    return image_loader
