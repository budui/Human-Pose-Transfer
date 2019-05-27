import os

import torch
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import dataset.bone_dataset as dataset
from loss.attr_loss import IDAttrLoss
from loss.mask_l1 import MaskL1Loss
from loss.perceptual_loss import PerceptualLoss
from models import PNGAN
from models.PG2 import weights_init_normal
from train.common_handler import warp_common_handler
from train.helper import move_data_pair_to
from util.arg_parse import bool_arg
from util.util import get_current_visuals

FAKE_IMG_FNAME = 'iteration_{}.png'
VAL_IMG_FNAME = 'train_image/epoch_{:02d}_{:07d}.png'


def _get_val_data_pairs(option, device):
    val_image_dataset = dataset.BoneDataset(
        os.path.join(option.market1501, "bounding_box_test/"),
        "data/market/test/pose_map_image/",
        "data/market/test/pose_mask_image/",
        option.test_pair_path,
        "data/market/annotation-test.csv",
    )
    val_image_loader = DataLoader(val_image_dataset, batch_size=8, num_workers=1, shuffle=True)
    val_data_pair = next(iter(val_image_loader))
    move_data_pair_to(device, val_data_pair)
    return val_data_pair


def get_trainer(opt, device="cuda"):
    G = PNGAN.ResGenerator(64, opt.num_res)
    D = PNGAN.PatchDiscriminator(64)
    D.apply(weights_init_normal)
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    lr_policy = lambda epoch: (1 - 1 * max(0, epoch - opt.de_epoch) / opt.de_epoch)
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_policy)
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_policy)

    if opt.l1_loss > 0:
        print("use l1_loss")
        l1_loss = nn.L1Loss().to(device)

    if opt.attr_loss > 0:
        print("use attr_loss")
        attr_loss = IDAttrLoss(opt.arp_path)

    if opt.mask_l1_loss > 0:
        print("use mask_l1_loss")
        mask_l1_loss = MaskL1Loss().to(device)

    if opt.perceptual_loss > 0:
        print("use perceptual_loss")
        perceptual_loss = PerceptualLoss(opt.perceptual_layers, device)
        print(perceptual_loss)

    gan_loss = nn.MSELoss().to(device)

    fake_loss = torch.zeros([1], device=device, requires_grad=False, dtype=torch.float)

    def step(engine, batch):
        move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        target_pose = batch["BP2"]
        target_img = batch["P2"]
        target_mask = batch["MP2"]

        generated_img = G(condition_img, target_pose)
        pred_real_g = D(generated_img, condition_img)
        g_adv_loss = gan_loss(pred_real_g, torch.ones_like(pred_real_g))

        if opt.mask_l1_loss > 0:
            g_mask_l1_loss = mask_l1_loss(generated_img, target_img, target_mask)
        else:
            g_mask_l1_loss = fake_loss

        if opt.l1_loss > 0:
            g_l1_loss = l1_loss(generated_img, target_img)
        else:
            g_l1_loss = fake_loss

        if opt.attr_loss > 0:
            g_attr_loss = attr_loss(generated_img, batch["attr"])
        else:
            g_attr_loss = fake_loss

        if opt.perceptual_loss > 0:
            g_perceptual_loss = perceptual_loss(generated_img, target_img)
        else:
            g_perceptual_loss = fake_loss

        g_loss = g_adv_loss + \
                 g_l1_loss * opt.l1_loss + \
                 g_mask_l1_loss * opt.mask_l1_loss + \
                 g_perceptual_loss * opt.perceptual_loss + \
                 g_attr_loss * opt.attr_loss

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        pred_fake_d = D(generated_img.detach(), condition_img)
        pred_real_d = D(target_img, condition_img)

        g_adv_real_loss = gan_loss(pred_real_d, torch.ones_like(pred_real_d))
        g_adv_fake_loss = gan_loss(pred_fake_d, torch.zeros_like(pred_fake_d))

        d_adv_loss = (g_adv_fake_loss + g_adv_real_loss) * 0.5

        optimizer_D.zero_grad()
        d_adv_loss.backward()
        optimizer_D.step()

        if engine.state.iteration % opt.print_freq == 0:
            path = os.path.join(opt.output_dir, VAL_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            get_current_visuals(path, batch, [generated_img])

        return {
            "pred": {
                # cause we do sigmoid in loss, here we must use sigmoid again.
                "G_real": torch.sigmoid(pred_real_g).mean().item(),
                "D_fake": torch.sigmoid(pred_fake_d).mean().item(),
                "D_real": torch.sigmoid(pred_real_d).mean().item()
            },
            "loss": {
                "G_adv": g_adv_loss.item(),
                "G_l1": g_l1_loss.item(),
                "G_per": g_perceptual_loss.item(),
                "G_ml1": g_mask_l1_loss.item(),
                "G_attr": g_attr_loss.item(),
                "G": g_loss.item(),
                "D": d_adv_loss.item()
            },
        }

    trainer = Engine(step)

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        print("-----------scheduler------step----------")
        print(optimizer_D.param_groups[0]["lr"])
        print(optimizer_G.param_groups[0]["lr"])
        scheduler_G.step()
        scheduler_D.step()
        print(optimizer_D.param_groups[0]["lr"])
        print(optimizer_G.param_groups[0]["lr"])
        print("-----------scheduler------over----------")

    # attach running average metrics
    monitoring_metrics = ['pred_G_real', 'pred_D_real', 'loss_G', 'loss_D']
    RunningAverage(output_transform=lambda x: x["pred"]['G_real']).attach(trainer, 'pred_G_real')
    RunningAverage(output_transform=lambda x: x["pred"]['D_fake']).attach(trainer, 'pred_D_fake')
    RunningAverage(output_transform=lambda x: x["pred"]['D_real']).attach(trainer, 'pred_D_real')

    RunningAverage(output_transform=lambda x: x["loss"]['G']).attach(trainer, 'loss_G')
    RunningAverage(output_transform=lambda x: x["loss"]['D']).attach(trainer, 'loss_D')
    RunningAverage(output_transform=lambda x: x["loss"]['G_adv']).attach(trainer, 'loss_G_adv')

    if opt.perceptual_loss:
        RunningAverage(output_transform=lambda x: x["loss"]['G_per']).attach(trainer, 'loss_G_per')
    if opt.mask_l1_loss:
        RunningAverage(output_transform=lambda x: x["loss"]['G_ml1']).attach(trainer, 'loss_G_ml1')
    if opt.l1_loss:
        RunningAverage(output_transform=lambda x: x["loss"]['G_l1']).attach(trainer, 'loss_G_l1')
    if opt.attr_loss:
        RunningAverage(output_transform=lambda x: x["loss"]['G_attr']).attach(trainer, 'loss_G_attr')

    networks_to_save = dict(G=G, D=D)

    def add_message(engine):
        message = " | G(total/adv): {:.3f}/{:.3f}".format(
            engine.state.metrics["loss_G"],
            engine.state.metrics["loss_G_adv"],
        )
        message += " | D(adv): {:.3f}".format(
            engine.state.metrics["loss_D"],
        )
        message += " | Pred(Gr/Df/Dr/): {:.3f}/{:.3f}/{:.3f}".format(
            engine.state.metrics["pred_G_real"],
            engine.state.metrics["pred_D_fake"],
            engine.state.metrics["pred_D_real"]
        )
        return message

    warp_common_handler(
        trainer,
        opt,
        networks_to_save,
        monitoring_metrics,
        add_message,
        [FAKE_IMG_FNAME, VAL_IMG_FNAME]
    )

    val_data_pairs = _get_val_data_pairs(opt, device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def save_example(engine):
        if engine.state.iteration > 0 and engine.state.iteration % opt.print_freq == 0:
            generated_img = G(val_data_pairs["P1"], val_data_pairs["BP2"])
            path = os.path.join(opt.output_dir, FAKE_IMG_FNAME.format(engine.state.iteration))
            get_current_visuals(path, val_data_pairs, [generated_img])

    return trainer


def get_data_loader(opt):
    image_dataset = dataset.AttrBoneDataset(
        "data/market/attribute/market_attribute.mat",
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


def add_new_arg_for_parser(parser):
    parser.add_argument('--market1501', type=str, default="../DataSet/Market-1501-v15.09.15/")
    parser.add_argument('--train_pair_path', type=str, default="data/market/pairs-train.csv")
    parser.add_argument('--test_pair_path', type=str, default="data/market/pairs-test.csv")
    parser.add_argument('--replacement', default=False, type=bool_arg)
    parser.add_argument('--flip_rate', type=float, default=0.5)

    parser.add_argument('--mask_l1_loss', type=float, default=10)
    parser.add_argument('--l1_loss', type=float, default=0)
    parser.add_argument('--perceptual_loss', type=float, default=10)
    parser.add_argument('--perceptual_layers', type=int, default=3,
                        help=" perceptual layers of perceptual_loss")
    parser.add_argument('--attr_loss', type=float, default=1)
    parser.add_argument('--arp_path', type=str, default="./data/net_ARP.pth")

    parser.add_argument('--num_res', type=int, default=9,
                        help="the number of res block in generator")
    parser.add_argument('--de_epoch', type=int, default=6,
                        help="the number of res block in generator")
