#!/usr/bin/env python3

import os
from argparse import ArgumentParser
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage


import dataset.bone_dataset as dataset
import models.PG2 as PG2
from util.v import get_current_visuals, get_current_visuals_
from loss.mask_l1 import MaskL1Loss
from util.image_pool import ImagePool


PRINT_FREQ = 50
FAKE_IMG_FNAME = 'epoch_{:04d}.png'
VAL_IMG_FNAME = 'train_img/epoch_{:04d}_{:04d}.png'
LOGS_FNAME = 'logs.tsv'
PLOT_FNAME = 'plot.svg'
CKPT_PREFIX = 'networks'


def move_data_pair_to(device, data_pair):
    # move data to GPU
    for k in data_pair:
        if "path" in k:
            # do not move string
            continue
        else:
            data_pair[k] = data_pair[k].to(device)


def get_stage_2_trainer(option, loader_size, val_data_pair, generator_1):
    generator_2 = PG2.G2(3 + 3, hidden_num=64, repeat_num=3, skip_connect=1)
    discriminator = PG2.NDiscriminator(in_channels=6)
    generator_1.to(device)
    generator_2.to(device)
    discriminator.to(device)

    optimizer_generator_2 = optim.Adam(generator_2.parameters(), lr=option.g_lr, betas=(option.beta1, option.beta2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=option.d_lr, betas=(option.beta1, option.beta2))

    bce_loss = nn.BCELoss().to(device)
    mask_l1_loss = MaskL1Loss().to(device)

    mask_l1_loss_lambda = option.mask_l1_loss_lambda
    batch_size = option.batch_size
    output_dir = option.output_dir
    epochs = option.stage_2_epoch_num

    real_labels = torch.ones((batch_size, 1), device=device)
    fake_labels = torch.zeros((batch_size, 1), device=device)

    fake_pair_img_pool = ImagePool(50)

    def step(engine, batch):
        move_data_pair_to(device, batch)
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
        pred_disc_fake_1 = discriminator(torch.cat([condition_img, generated_img], dim=1))
        generator_2_bce_loss = bce_loss(pred_disc_fake_1, real_labels)
        # MaskL1 loss
        generator_2_mask_l1_loss = mask_l1_loss(generated_img, target_img, target_mask)
        # total loss
        generator_2_loss = generator_2_bce_loss + mask_l1_loss_lambda * generator_2_mask_l1_loss
        # gradient update
        generator_2_loss.backward()
        optimizer_generator_2.step()

        # -----------------------------------------------------------
        # (2) Update D network: minimize L_bce
        optimizer_discriminator.zero_grad()
        # real loss
        real_pair_img = torch.cat([condition_img, target_img], dim=1)
        pred_disc_real_2 = discriminator(real_pair_img)
        discriminator_real_loss = bce_loss(pred_disc_real_2, real_labels)
        # fake loss
        fake_pair_img = torch.cat([condition_img, generated_img], dim=1)
        #fake_pair_img = fake_pair_img_pool.query(torch.cat([condition_img, generated_img], dim=1).data)
        pred_disc_fake_2 = discriminator(fake_pair_img.detach())
        discriminator_fake_loss = bce_loss(pred_disc_fake_2, fake_labels)
        # total loss
        discriminator_loss = (discriminator_fake_loss + discriminator_real_loss) * 0.5
        discriminator_loss.backward()
        # gradient update
        optimizer_discriminator.step()

        # -----------------------------------------------------------
        # (3) Collect train info

        if engine.state.iteration % 100 == 0:
            path = os.path.join(output_dir, VAL_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            get_current_visuals_(path, batch, [generator_1_img, diff_img, generated_img])

        return {
            "pred": {
                "G_fake": pred_disc_fake_1.mean().item(),
                "D_fake": pred_disc_fake_2.mean().item(),
                "D_real": pred_disc_real_2.mean().item()
            },
            "loss": {
                "G_bce": generator_2_bce_loss.item(),
                "G_l1": generator_2_mask_l1_loss.item(),
                "G": generator_2_loss.item(),
                "D_real": discriminator_real_loss.item(),
                "D_fake": discriminator_fake_loss.item(),
                "D": discriminator_loss.item()
            },
        }

        # ignite objects
    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(output_dir, CKPT_PREFIX, save_interval=2, n_saved=5, require_empty=False)

    timer = Timer(average=True)

    # attach running average metrics
    monitoring_metrics = ['pred_D_fake', 'pred_D_real', 'loss_G',  'loss_D']
    RunningAverage(output_transform=lambda x: x["pred"]['G_fake']).attach(trainer, 'pred_G_fake')
    RunningAverage(output_transform=lambda x: x["pred"]['D_fake']).attach(trainer, 'pred_D_fake')
    RunningAverage(output_transform=lambda x: x["pred"]['D_real']).attach(trainer, 'pred_D_real')

    RunningAverage(output_transform=lambda x: x["loss"]['G']).attach(trainer, 'loss_G')
    RunningAverage(output_transform=lambda x: x["loss"]['G_bce']).attach(trainer, 'loss_G_bce')
    RunningAverage(output_transform=lambda x: x["loss"]['G_l1']).attach(trainer, 'loss_G_l1')

    RunningAverage(output_transform=lambda x: x["loss"]['D']).attach(trainer, 'loss_D')
    RunningAverage(output_transform=lambda x: x["loss"]['D_real']).attach(trainer, 'loss_D_real')
    RunningAverage(output_transform=lambda x: x["loss"]['D_fake']).attach(trainer, 'loss_D_fake')

    # attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % PRINT_FREQ == 0:
            fname = os.path.join(output_dir, LOGS_FNAME)
            columns = sorted(engine.state.metrics.keys())
            values = [str(round(engine.state.metrics[value], 5)) for value in columns]

            with open(fname, 'a') as f:
                if f.tell() == 0:
                    print('\t'.join(columns), file=f)
                print('\t'.join(values), file=f)

            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(epoch=engine.state.epoch,
                                                                  max_epoch=epochs,
                                                                  i=(engine.state.iteration % loader_size),
                                                                  max_i=loader_size)

            message += " | G_loss(all/bce/l1): {:.4f}/{:.4f}/{:.4f}".format(
                engine.state.metrics["loss_G"],
                engine.state.metrics["loss_G_bce"],
                engine.state.metrics["loss_G_l1"]
            )
            message += " | D_loss(all/fake/real): {:.4f}/{:.4f}/{:.4f}".format(
                engine.state.metrics["loss_D"],
                engine.state.metrics["loss_D_fake"],
                engine.state.metrics["loss_D_real"]
            )
            message += " | Pred(G2_fake/D_fake/D_real/): {:.4f}/{:.4f}/{:.4f}".format(
                engine.state.metrics["pred_G_fake"],
                engine.state.metrics["pred_D_fake"],
                engine.state.metrics["pred_D_real"]
            )

            pbar.log_message(message)

    # adding handlers using `trainer.add_event_handler` method API
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={
                                  'netG2': generator_2,
                                  'netD': discriminator
                              })
    # automatically adding handlers via a special `attach` method of `Timer` handler
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.STARTED)
    def mkdir(engine):
        print("--------------------------------")
        os.mkdir(os.path.join(output_dir, "train_img"))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
        timer.reset()

        # adding handlers using `trainer.on` decorator API

    @trainer.on(Events.EPOCH_COMPLETED)
    def create_plots(engine):
        try:
            import matplotlib as mpl
            mpl.use('agg')

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

        except ImportError:
            warnings.warn('Loss plots will not be generated -- pandas or matplotlib not found')

        else:
            df = pd.read_csv(os.path.join(output_dir, LOGS_FNAME), delimiter='\t')
            #x = np.arange(1, engine.state.epoch * engine.state.iteration + 1, PRINT_FREQ)
            _ = df.plot(subplots=True, figsize=(10, 10))
            _ = plt.xlabel('Iteration number')
            fig = plt.gcf()
            path = os.path.join(output_dir, PLOT_FNAME)

            fig.savefig(path)
            fig.clear()

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            create_plots(engine)
            checkpoint_handler(engine, {
                'netG2_exception': generator_2,
                'netD_exception': generator_1
            })

        else:
            raise e

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_example(engine):
        img_g1 = generator_1(torch.cat([val_data_pair["P1"], val_data_pair["BP2"]], dim=1))
        diff_map = generator_2(torch.cat([val_data_pair["P1"], img_g1], dim=1))
        img_g2 = diff_map + img_g1

        path = os.path.join(output_dir, FAKE_IMG_FNAME.format(engine.state.epoch))
        get_current_visuals_(path, val_data_pair, [img_g1,diff_map,img_g2])

    return trainer


def main(option):
    image_dataset = dataset.BoneDataset(
        "../DataSet/Market-1501-v15.09.15/bounding_box_train/",
        "data/market/train/pose_map_image/",
        "data/market/train/pose_mask_image/",
        "data/market-pairs-train.csv",
        random_select=True
    )
    print(image_dataset)

    image_loader = DataLoader(image_dataset, batch_size=opt.batch_size, num_workers=8, pin_memory=True)

    val_image_dataset = dataset.BoneDataset(
        "../DataSet/Market-1501-v15.09.15/bounding_box_test/",
        "data/market/test/pose_map_image/",
        "data/market/test/pose_mask_image/",
        "data/market-pairs-test.csv",
        random_select=True,
        random_select_size=5
    )

    val_image_loader = DataLoader(val_image_dataset, batch_size=4, num_workers=1)
    data_pair = next(iter(val_image_loader))
    move_data_pair_to(device, data_pair)

    generator_1 = PG2.G1(3 + 18, repeat_num=5, half_width=True, middle_z_dim=64)
    generator_1.load_state_dict(torch.load("checkpoints/G1.pth"))

    stage_2_trainer = get_stage_2_trainer(option, len(image_loader), data_pair, generator_1)

    stage_2_trainer.run(image_loader, max_epochs=option.stage_2_epoch_num)


if __name__ == '__main__':
    parser = ArgumentParser(description='Training')
    parser.add_argument('--gpu_id', default=2, type=int, help='gpu_id: e.g. 0')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument("--stage_1_epoch_num", default=100, type=int, help="stage_1_epoch_num")
    parser.add_argument("--stage_2_epoch_num", default=80, type=int, help="stage_2_epoch_num")
    parser.add_argument('--d_lr', type=float, default=0.00002)
    parser.add_argument('--g_lr', type=float, default=0.00002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--mask_l1_loss_lambda', type=float, default=10)
    parser.add_argument("--output_dir", type=str, default="ckp/")

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    cudnn.benchmark = True

    device = "cuda"

    main(opt)
