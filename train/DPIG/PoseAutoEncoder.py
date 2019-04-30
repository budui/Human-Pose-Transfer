import os
import itertools

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from dataset.key_point_dataset import KeyPointDataset
from models.DPIG import PoseDecoder, PoseEncoder
from util.vis.pose import show as show_pose
from train.common_handler import warp_common_handler


PRINT_FREQ = 200
FAKE_IMG_FNAME = 'epoch_{:04d}.png'
VAL_IMG_FNAME = 'train_img/epoch_{:04d}_{:04d}.png'
LOGS_FNAME = 'loss.log'
PLOT_FNAME = 'plot.svg'
CKPT_PREFIX = 'networks'


def get_trainer(option, device):
    pose_decoder = PoseDecoder()
    pose_encoder = PoseEncoder()
    pose_decoder.to(device)
    pose_encoder.to(device)

    output_dir = option.output_dir

    l2_loss = nn.MSELoss()
    l2_loss.to(device)

    op_auto_encoder = optim.Adam(
        itertools.chain(pose_decoder.parameters(), pose_encoder.parameters()),
        lr=option.lr, betas=(option.beta1, option.beta2)
    )

    def step(engine, origin_pose):
        origin_pose = origin_pose.to(device)

        op_auto_encoder.zero_grad()

        hide_z = pose_encoder(origin_pose)
        recon_pose = pose_decoder(hide_z)

        recon_loss = l2_loss(recon_pose, origin_pose)
        recon_loss.backward()
        op_auto_encoder.step()

        return {
            "recon_loss": recon_loss.item(),
            "recon_pose": recon_pose[0],
            "origin_pose": origin_pose[0]
        }

    trainer = Engine(step)

    RunningAverage(output_transform=lambda x: x["recon_loss"]).attach(trainer, 'loss')

    def add_message(engine):
        message = " | loss: {:.4f}".format(
                engine.state.metrics["loss"],
            )
        return message

    networks_to_save = dict(pose_encoder=pose_encoder, pose_decoder=pose_decoder)

    warp_common_handler(
        trainer,
        option,
        networks_to_save,
        ["loss"],
        add_message,
        [VAL_IMG_FNAME]
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def save_train_pose(engine):
        if (engine.state.iteration - 1) % PRINT_FREQ == 0:
            show_pose([
                engine.state.output["origin_pose"],
                engine.state.output["recon_pose"]
                ],
                os.path.join(output_dir, VAL_IMG_FNAME.format(
                    engine.state.epoch, engine.state.iteration
                ))
            )
    return trainer


def add_new_arg_for_parser(parser):
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument("--key_points_dir", type=str,
                        default="data/market/annotation-train.csv")


def get_data_loader(option):
    print("loading dataset ...")
    dataset = KeyPointDataset(option.key_points_dir)
    print(dataset)
    loader = DataLoader(dataset, batch_size=option.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    return loader


if __name__ == '__main__':
    pass