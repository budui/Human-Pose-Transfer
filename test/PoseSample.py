import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

from dataset.key_point_dataset import KeyPointDataset
from models.DPIG import PoseDecoder, PoseEncoder
from util.vis.pose import show as show_pose

IMG_FNAME = 'iter_{iter}.jpg'


def _load_model(model_class, model_save_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    model.eval()
    return model


def get_tester(option, device):
    pose_encoder = _load_model(PoseEncoder, option.encoder_path, device)
    pose_decoder = _load_model(PoseDecoder, option.decoder_path, device)

    l2_loss = nn.MSELoss()
    l2_loss.to(device)

    output_dir = option.output_dir

    def step(engine, origin_pose):
        origin_pose = origin_pose.to(device)

        z = pose_encoder(origin_pose)
        recon_pose = pose_decoder(z)

        recon_loss = l2_loss(recon_pose, origin_pose)

        return {
            "recon_loss": recon_loss.item(),
            "recon_pose": recon_pose,
            "origin_pose": origin_pose,
            "z": z
        }

    tester = Engine(step)

    RunningAverage(output_transform=lambda x: x["recon_loss"]).attach(tester, 'loss')
    pbar = ProgressBar()
    pbar.attach(tester, metric_names=["loss"])

    @tester.on(Events.ITERATION_COMPLETED)
    def save_result(engine):
        show_pose([
            engine.state.output["origin_pose"],
            engine.state.output["recon_pose"]
        ],
            os.path.join(output_dir, IMG_FNAME.format(iter=engine.state.iteration))
        )

    @tester.on(Events.STARTED)
    def mkdir(engine):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    return tester


def add_new_arg_for_parser(parser):
    parser.add_argument("--key_points_dir", type=str,
                        default="data/market/annotation-test.csv")
    parser.add_argument("--encoder_path", type=str)
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument("--limit_size", type=int, help="if set gen_size, only `limit_size` poses will be used")


def get_data_loader(option):
    print("loading dataset ...")
    dataset = KeyPointDataset(option.key_points_dir)
    print(dataset)
    if option.limit_size is not None:
        dataset = Subset(dataset, list(range(option.limit_size)))
        print("only use {} poses".format(option.limit_size))
    loader = DataLoader(dataset, batch_size=option.batch_size, num_workers=8, pin_memory=True)
    return loader
