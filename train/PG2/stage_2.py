import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage


import dataset.bone_dataset as dataset
import models.PG2 as PG2
from util.util import get_current_visuals
from loss.mask_l1 import MaskL1Loss
from loss.perceptual_loss import PerceptualLoss
from util.image_pool import ImagePool
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

    val_image_loader = DataLoader(val_image_dataset, batch_size=8, num_workers=1, shuffle=True)
    val_data_pair = next(iter(val_image_loader))
    _move_data_pair_to(device, val_data_pair)

    generator_1 = PG2.G1(3 + 18, repeat_num=5, half_width=True, middle_z_dim=64)
    generator_1.load_state_dict(torch.load(option.G1_path))
    generator_2 = PG2.G2(3 + 3, hidden_num=64, repeat_num=3, skip_connect=1)
    discriminator = PG2.Discriminator(in_channels=3)
    generator_1.to(device)
    generator_2.to(device)
    discriminator.to(device)

    optimizer_generator_2 = optim.Adam(generator_2.parameters(), lr=option.g_lr, betas=(option.beta1, option.beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=option.d_lr, betas=(option.beta1, option.beta2))
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_generator_2, step_size=1, gamma=0.8)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_discriminator, step_size=1, gamma=0.8)
    mask_l1_loss_lambda = option.mask_l1_loss_lambda
    if mask_l1_loss_lambda > 0:
        print("using mask L1Loss weights: {}".format(mask_l1_loss_lambda))
        mask_l1_loss = MaskL1Loss().to(device)

    perceptual_loss_lambda = option.perceptual_loss_lambda
    if perceptual_loss_lambda > 0:
        print("using PerceptualLoss. weights: {}".format(perceptual_loss_lambda))
        perceptual_loss = PerceptualLoss(device=device).to(device)
        print(perceptual_loss)

    bce_loss = nn.BCELoss().to(device)
    bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)

    batch_size = option.batch_size
    output_dir = option.output_dir

    real_labels = torch.ones((batch_size, 1), device=device)
    fake_labels = torch.zeros((batch_size, 1), device=device)
    fake_loss = torch.zeros([1], device=device, requires_grad=False, dtype=torch.float)

    def step(engine, batch):
        _move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        condition_pose = batch["BP2"]
        target_img = batch["P2"]
        target_mask = batch["MP2"]

        # get generated img
        generator_1_img = generator_1(torch.cat([condition_img, condition_pose], dim=1))
        diff_img = generator_2(torch.cat([condition_img, generator_1_img], dim=1))
        generated_img = generator_1_img + diff_img

        # -----------------------------------------------------------
        # (1) Update G2 network: minimize L_bce + L_1
        optimizer_generator_2.zero_grad()

        # BCE loss
        pred_disc_fake_1 = discriminator(generated_img)
        generator_2_bce_loss = bce_with_logits_loss(pred_disc_fake_1, real_labels)
        # MaskL1 loss
        if mask_l1_loss_lambda > 0:
            generator_2_mask_l1_loss = mask_l1_loss(generated_img, target_img, target_mask)
        else:
            generator_2_mask_l1_loss = fake_loss
        # Perceptual loss
        if perceptual_loss_lambda > 0:
            generator_2_perceptual_loss = perceptual_loss(generated_img, target_img)
        else:
            generator_2_perceptual_loss = fake_loss
        # total loss
        generator_2_loss = generator_2_bce_loss + \
                           mask_l1_loss_lambda * generator_2_mask_l1_loss + \
                           perceptual_loss_lambda * generator_2_perceptual_loss
        # gradient update
        generator_2_loss.backward()
        optimizer_generator_2.step()

        # -----------------------------------------------------------
        # (2) Update D network: minimize L_bce
        optimizer_discriminator.zero_grad()
        # real loss 1
        pred_disc_real_2 = discriminator(target_img)
        discriminator_real_loss = bce_with_logits_loss(pred_disc_real_2, real_labels)
        # fake loss 1
        pred_disc_fake_2 = discriminator(generated_img.detach())
        discriminator_fake_loss = bce_with_logits_loss(pred_disc_fake_2, fake_labels)
        # total loss 1
        discriminator_loss_1 = (discriminator_fake_loss + discriminator_real_loss) * 0.5

        # real loss 2
        pred_disc_real_2 = discriminator(target_img)
        discriminator_real_loss = bce_with_logits_loss(pred_disc_real_2, real_labels)
        # fake loss 2
        pred_disc_fake_2 = discriminator(condition_img)
        discriminator_fake_loss = bce_with_logits_loss(pred_disc_fake_2, fake_labels)
        discriminator_loss_2 = (discriminator_fake_loss + discriminator_real_loss) * 0.5

        discriminator_loss = (discriminator_loss_1 + discriminator_loss_2)*0.5
        discriminator_loss.backward()
        # gradient update
        optimizer_discriminator.step()

        # -----------------------------------------------------------
        # (3) Collect train info

        if engine.state.iteration % 100 == 0:
            path = os.path.join(output_dir, VAL_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            get_current_visuals(path, batch, [generator_1_img, diff_img, generated_img])

        return {
            "pred": {
                # cause we do sigmoid in loss, here we must use sigmoid again.
                "G_fake": torch.sigmoid(pred_disc_fake_1).mean().item() ,
                "D_fake": torch.sigmoid(pred_disc_fake_2).mean().item(),
                "D_real": torch.sigmoid(pred_disc_real_2).mean().item()
            },
            "loss": {
                "G_bce": generator_2_bce_loss.item(),
                "G_per": generator_2_perceptual_loss.item(),
                "G_l1": generator_2_mask_l1_loss.item(),
                "G": generator_2_loss.item(),
                "D_real": discriminator_real_loss.item(),
                "D_fake": discriminator_fake_loss.item(),
                "D": discriminator_loss.item()
            },
        }

        # ignite objects

    trainer = Engine(step)

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        print("-----------scheduler------step----------")
        print(optimizer_discriminator.param_groups[0]["lr"])
        print(optimizer_generator_2.param_groups[0]["lr"])
        scheduler_g.step()
        scheduler_d.step()
        print(optimizer_discriminator.param_groups[0]["lr"])
        print(optimizer_generator_2.param_groups[0]["lr"])
        print("-----------scheduler------over----------")

    # attach running average metrics
    monitoring_metrics = ['pred_G_fake', 'pred_D_real', 'loss_G',  'loss_D']
    RunningAverage(output_transform=lambda x: x["pred"]['G_fake']).attach(trainer, 'pred_G_fake')
    RunningAverage(output_transform=lambda x: x["pred"]['D_fake']).attach(trainer, 'pred_D_fake')
    RunningAverage(output_transform=lambda x: x["pred"]['D_real']).attach(trainer, 'pred_D_real')

    RunningAverage(output_transform=lambda x: x["loss"]['G']).attach(trainer, 'loss_G')
    RunningAverage(output_transform=lambda x: x["loss"]['G_bce']).attach(trainer, 'loss_G_bce')
    RunningAverage(output_transform=lambda x: x["loss"]['G_l1']).attach(trainer, 'loss_G_l1')
    RunningAverage(output_transform=lambda x: x["loss"]['G_per']).attach(trainer, 'loss_G_per')

    RunningAverage(output_transform=lambda x: x["loss"]['D']).attach(trainer, 'loss_D')
    RunningAverage(output_transform=lambda x: x["loss"]['D_real']).attach(trainer, 'loss_D_real')
    RunningAverage(output_transform=lambda x: x["loss"]['D_fake']).attach(trainer, 'loss_D_fake')

    networks_to_save = dict(G2=generator_2, D=discriminator)

    def add_message(engine):
        message = " | G(a/b/p/l): {:.3f}/{:.3f}/{:.3f}/{:.3f}".format(
            engine.state.metrics["loss_G"],
            engine.state.metrics["loss_G_bce"],
            engine.state.metrics["loss_G_per"],
            engine.state.metrics["loss_G_l1"]
        )
        message += " | D(a/f/r): {:.3f}/{:.3f}/{:.3f}".format(
            engine.state.metrics["loss_D"],
            engine.state.metrics["loss_D_fake"],
            engine.state.metrics["loss_D_real"]
        )
        message += " | Pred(Gf/Df/Dr/): {:.3f}/{:.3f}/{:.3f}".format(
            engine.state.metrics["pred_G_fake"],
            engine.state.metrics["pred_D_fake"],
            engine.state.metrics["pred_D_real"]
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
        if engine.state.iteration > 0 and engine.state.iteration % option.print_freq == 0:
            img_g1 = generator_1(torch.cat([val_data_pair["P1"], val_data_pair["BP2"]], dim=1))
            diff_map = generator_2(torch.cat([val_data_pair["P1"], img_g1], dim=1))
            img_g2 = diff_map + img_g1
            path = os.path.join(output_dir, FAKE_IMG_FNAME.format(engine.state.iteration))
            get_current_visuals(path, val_data_pair, [img_g1, diff_map, img_g2])

    return trainer


def add_new_arg_for_parser(parser):
    parser.add_argument('--d_lr', type=float, default=0.00002)
    parser.add_argument('--g_lr', type=float, default=0.00002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--replacement', action="store_true")
    parser.add_argument('--mask_l1_loss_lambda', type=float, default=10)
    parser.add_argument('--flip_rate', type=float, default=0.0)
    parser.add_argument('--perceptual_loss_lambda', type=float, default=10)
    parser.add_argument('--G1_path', type=str, default="checkpoints/G1.pth")
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
        flip_rate=opt.flip_rate,
    )

    image_loader = DataLoader(
        image_dataset, batch_size=opt.batch_size,
        num_workers=8, pin_memory=True, drop_last=True,
        sampler=torch.utils.data.RandomSampler(
            image_dataset,
            replacement=opt.replacement
        ),
    )
    print("dataset: {} num_batches: {}".format(image_dataset, len(image_loader)))
    return image_loader