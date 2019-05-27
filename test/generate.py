import json
import os
from shutil import copyfile

import numpy as np
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from torch.utils.data import DataLoader

import util.util as util
from dataset.bone_dataset import BoneDataset


def select_generator(option, device):
    if option.name == "PG2-Generate":
        from test.PG2 import get_generator
        return get_generator(option.G1_path, option.G2_path, device)
    elif option.name == "PNGAN-Generate":
        from test.PNGAN import get_generator
        return get_generator(option.g_path, option.num_res, device, option.show_all)
    else:
        raise NotImplementedError("not implemented generate methods: {}".format(option.name))


def get_tester(option, device):
    output_dir = option.output_dir
    generate = select_generator(option, device)

    limit = option.limit

    def step(engine, batch):
        generated_imgs = generate(batch)
        condition_names = batch["P1_path"]
        target_names = batch["P2_path"]

        for i in range(generated_imgs.size(0)):
            # image height and width
            # image_size = (generated_imgs.size(2), generated_imgs.size(3))
            image_size = (128, 64)
            image = np.zeros((image_size[0], image_size[1] * 2 + generated_imgs.size(3), 3)).astype(np.uint8)
            image[:, 0 * image_size[1]:1 * image_size[1], :] = util.tensor2image(batch["P1"].data[i])
            image[:, 1 * image_size[1]:2 * image_size[1], :] = util.tensor2image(batch["P2"].data[i])
            image[:, 2 * image_size[1]:, :] = util.tensor2image(generated_imgs.data[i])

            if limit < 0:
                image_path = os.path.join(output_dir, "{}___{}_vis.jpg".format(condition_names[i], target_names[i]))
            else:
                image_path = os.path.join(output_dir, "{}.png".format(engine.state.idx))
                engine.state.idx += 1
            util.save_image(image, image_path)
        return

    tester = Engine(step)

    pbar = ProgressBar()
    pbar.attach(tester)

    @tester.on(Events.STARTED)
    def mkdir(engine):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @tester.on(Events.STARTED)
    def show(engine):
        if limit > 0:
            engine.state.idx = 1
            copyfile("./util/compare_result.html", os.path.join(output_dir, "index.html"))
            with open(os.path.join(output_dir, "data.json"), "w") as data_f:
                json.dump({"limit": option.limit}, data_f)

    return tester


def add_new_arg_for_parser(parser, name):
    parser.add_argument("--pair_path", type=str, default="data/market/pairs-test.csv")
    parser.add_argument('--market1501', type=str, default="../dataset/Market-1501-v15.09.15/")
    parser.add_argument('--limit', default=-1, type=int, help='generated images amount limit. default is -1')
    if name == "PG2-Generate":
        parser.add_argument("--G1_path", type=str, default="./data/market/models/PG2/G1.pth")
        parser.add_argument("--G2_path", type=str)

    elif name == "PNGAN-Generate":
        parser.add_argument("--g_path", type=str)
        parser.add_argument("--show_all", type=bool, action="store_true")
        parser.add_argument('--num_res', type=int, default=9,
                            help="the number of res block in generator")


def get_data_loader(opt):
    image_dataset = BoneDataset(
        os.path.join(opt.market1501, "bounding_box_test/"),
        "data/market/test/pose_map_image/",
        "data/market/test/pose_mask_image/",
        opt.pair_path,
        "data/market/annotation-test.csv",
    )

    def generate_predictable_indices(limit):
        import numpy as np
        np.random.seed(520)
        arr = np.arange(len(image_dataset))
        np.random.shuffle(arr)
        return arr[:limit]

    print("load test dataset: {} pairs".format(len(image_dataset)))
    if opt.limit > 0:
        image_dataset = torch.utils.data.Subset(image_dataset, generate_predictable_indices(opt.limit))
    image_loader = DataLoader(image_dataset, batch_size=opt.batch_size, num_workers=8)
    return image_loader
