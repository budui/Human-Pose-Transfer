import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from ignite.engine import Engine, Events

from ignite.metrics import RunningAverage


import dataset.bone_dataset as dataset
import models.PG2 as PG2
from util.util import get_current_visuals
from loss.mask_l1 import MaskL1Loss
from train.common_handler import warp_common_handler

FAKE_IMG_FNAME = 'iteration_{}.png'
VAL_IMG_FNAME = 'train_image/epoch_{:02d}_{:07d}.png'

def _move_data_pair_to(device, data_pair):
    # move data to GPU
    for k in data_pair:
        if "path" in k:
            # do not move string
            continue
        else:
            data_pair[k] = data_pair[k].to(device)


def get_trainer(option, device):
    val_image_dataset = dataset.BoneDataset(
        os.path.join(option.market1501, "bounding_box_test/"),
        "data/market/test/pose_map_image/",
        "data/market/test/pose_mask_image/",
        option.test_pair_path,
        "data/market/annotation-test.csv",
    )

    val_image_loader = DataLoader(val_image_dataset, batch_size=4, num_workers=1)
    val_data_pair = next(iter(val_image_loader))
    _move_data_pair_to(device, val_data_pair)

    generator_1 = PG2.G1(3 + 18, repeat_num=5, half_width=True, middle_z_dim=64)
    generator_1.to(device)
    optimizer_generator_1 = optim.Adam(generator_1.parameters(), lr=option.g_lr, betas=(option.beta1, option.beta2))

    mask_l1_loss = MaskL1Loss().to(device)

    output_dir = option.output_dir

    def step(engine, batch):
        _move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        condition_pose = batch["BP2"]
        target_img = batch["P2"]
        target_mask = batch["MP2"]

        generator_1_img = generator_1(torch.cat([condition_img, condition_pose], dim=1))

        optimizer_generator_1.zero_grad()
        generator_1_mask_l1_loss = mask_l1_loss(generator_1_img, target_img, target_mask)
        generator_1_mask_l1_loss.backward()
        optimizer_generator_1.step()

        if engine.state.iteration % 100 == 0:
            path = os.path.join(output_dir, VAL_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            get_current_visuals(path, batch, [generator_1_img])

        return {
            "loss": {
                "G_l1": generator_1_mask_l1_loss.item(),
            },
        }

    trainer = Engine(step)

    # attach running average metrics
    monitoring_metrics = ['loss_G_l1']
    RunningAverage(output_transform=lambda x: x["loss"]['G_l1']).attach(trainer, 'loss_G_l1')

    networks_to_save = dict(G2=generator_1)

    def add_message(engine):
        message = " | G_loss: {:.4f}".format(
            engine.state.metrics["loss_G_l1"]
        )
        return message

    warp_common_handler(
        trainer,
        option,
        networks_to_save,
        monitoring_metrics,
        add_message,
        [FAKE_IMG_FNAME, VAL_IMG_FNAME]
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def save_example(engine):
        if (engine.state.iteration - 1) % option.print_freq == 0:
            img_g1 = generator_1(torch.cat([val_data_pair["P1"], val_data_pair["BP2"]], dim=1))
            path = os.path.join(output_dir, FAKE_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            get_current_visuals(path, val_data_pair, [img_g1])
    return trainer


def add_new_arg_for_parser(parser):
    parser.add_argument('--d_lr', type=float, default=0.00002)
    parser.add_argument('--g_lr', type=float, default=0.00002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--market1501', type=str, default="../DataSet/Market-1501-v15.09.15/")
    parser.add_argument('--train_pair_path', type=str, default="data/market/pairs-train.csv")
    parser.add_argument('--test_pair_path', type=str, default="data/market/pairs-test.csv")


def get_data_loader(opt):
    image_dataset = dataset.BoneDataset(
        os.path.join(opt.market1501, "bounding_box_train/"),
        "data/market/train/pose_map_image/",
        "data/market/train/pose_mask_image/",
        opt.train_pair_path,
        "data/market/annotation-train.csv",
        flip_rate=0.5,
    )
    print(image_dataset)
    image_loader = DataLoader(image_dataset, batch_size=opt.batch_size,
                              num_workers=8, pin_memory=True,
                              drop_last=True, shuffle=True
                              )
    return image_loader
