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
from util.arg_parse import bool_arg


from util.sample.pose import interpolation
from test.DPIG.sample_pose import _load_model
from util.util import show_with_visibility as show_pose
from models.DPIG import PoseDecoder, PoseEncoder
from dataset.key_point_dataset import KeyPointDataset


def select_generator(option, device):
    if option.name == "PG2-Generate":
        from test.PG2 import get_generator
        return get_generator(option.G1_path, option.G2_path, device)
    elif option.name == "PNGAN-Generate":
        from test.PNGAN import get_generator
        return get_generator(option.g_path, option.num_res, device, option.show_all)
    elif option.name == "PAGAN-Generate":
        from test.PAGAN import get_generator
        return get_generator(option.g_path, device)
    else:
        raise NotImplementedError("not implemented generate methods: {}".format(option.name))


def get_interpolation_pose():
    device = "cuda"
    encoder_path = "/root/hpt/data/pose_encoder.pth"
    decoder_path = "/root/hpt/data/pose_decoder.pth"
    pose_encoder = _load_model(PoseEncoder, encoder_path, device)
    pose_decoder = _load_model(PoseDecoder, decoder_path, device)
    print("################# load pose encoder and decoder #######################")

    key_points_dir = "/root/hpt/data/market/annotation-test.csv"
    dataset = KeyPointDataset(key_points_dir)

    def inter(image1, image2):
        pose_1, pose_2 = dataset.get(image1, image2)
        pose_1 = pose_1.to("cuda")
        pose_2 = pose_2.to("cuda")
        pose_interpolation = interpolation(pose_1, pose_2, pose_encoder, pose_decoder)
        new_poses = [pose.squeeze(0) for pose in pose_interpolation]
        return new_poses
    return inter


def get_tester(option, device):
    output_dir = option.output_dir
    generate = select_generator(option, device)

    limit = option.limit

    inter = get_interpolation_pose()

    def step(engine, batch):

        i = True
        if i:
            new_poses = inter(batch["P1_name"][0], batch["P2_name"][0])
            show_pose(new_poses, "{}_{}_pose.jpg".format(batch["P1_name"][0], batch["P2_name"][0]))
            new_poses
            generated_imgs = generate(batch)
        else:
            generated_imgs = generate(batch)

        if limit < 0:
            util.visuals_for_test(output_dir, batch, generated_imgs)
        else:
            util.visuals_for_test(output_dir, batch, generated_imgs, name=engine.state.idx)
            engine.state.idx += generated_imgs.size(0)
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
            copyfile("tool/html/show_generated.html", os.path.join(output_dir, "index.html"))
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
        parser.add_argument('--show_all', default=False, type=bool_arg)
        parser.add_argument('--num_res', type=int, default=9, help="the number of res block in generator")
    elif name == "PAGAN-Generate":
        parser.add_argument("--g_path", type=str)


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
        np.random.seed(252)
        arr = np.arange(len(image_dataset))
        np.random.shuffle(arr)
        return arr[:limit]

    print("load test dataset: {} pairs".format(len(image_dataset)))
    if opt.limit > 0:
        image_dataset = torch.utils.data.Subset(image_dataset, generate_predictable_indices(opt.limit))
    image_loader = DataLoader(image_dataset, batch_size=opt.batch_size, num_workers=8)
    return image_loader
