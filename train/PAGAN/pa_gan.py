import os

import torch
import torch.optim as optim
from ignite.engine import Engine, Events
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import dataset.bone_dataset as dataset
from loss.attr_loss import IDAttrLoss
from loss.mask_l1 import MaskL1Loss
from loss.perceptual_loss import PerceptualLoss
from models import PNGAN, PATN, PAGAN
from models.PG2 import weights_init_normal
from train.common_handler import warp_common_handler
from train.helper import move_data_pair_to, LossContainer, attach_engine
from util.arg_parse import bool_arg
from util.util import get_current_visuals
from util.image_pool import ImagePool

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
    G = PAGAN.PAGenerator()
    D = PNGAN.PatchDiscriminator(64)
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    G.to(device)
    D.to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    lr_policy = lambda epoch: (1 - 1 * max(0, epoch - opt.de_epoch) / opt.de_epoch)
    scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_policy)
    scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_policy)

    if opt.use_db:
        DB = PATN.ResnetDiscriminator(3 + 18, gpu_ids=[opt.gpu_id], n_blocks=3, use_sigmoid=False)
        DB.apply(weights_init_normal)
        DB.to(device)
        optimizer_DB = optim.Adam(DB.parameters(), lr=0.0002, betas=(0.5, 0.999))
        scheduler_DB = lr_scheduler.LambdaLR(optimizer_DB, lr_lambda=lr_policy)
        fake_pb_pool = ImagePool(50)

    l1_loss = LossContainer(nn.L1Loss(), opt.l1_loss)
    mask_l1_loss = LossContainer(MaskL1Loss(), opt.mask_l1_loss)
    attr_loss = LossContainer(IDAttrLoss(opt.arp_path), opt.attr_loss)
    perceptual_loss = LossContainer(PerceptualLoss(opt.perceptual_layers, device), opt.perceptual_loss)
    gan_loss = nn.MSELoss()

    def step(engine, batch):
        move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        target_pose = batch["BP2"]
        target_img = batch["P2"]
        target_mask = batch["MP2"]

        generated_img = G(condition_img, torch.cat([batch["BP1"], batch["BP2"]], dim=1))

        pred = {"g_pp": D(generated_img, condition_img)}

        _generator_loss = {
            "adv":  gan_loss(pred["g_pp"], torch.ones_like(pred["g_pp"])),
            "mask_l1": mask_l1_loss(generated_img, target_img, target_mask),
            "l1": l1_loss(generated_img, target_img),
            "attr": attr_loss(generated_img, batch["attr"], batch["P1_path"]),
            "perceptual": perceptual_loss(generated_img, target_img)
        }

        if opt.use_db:
            pred["g_pb"] = DB(torch.cat([generated_img, target_pose], dim=1))
            _generator_loss["adv_pb"] = gan_loss(pred["g_pb"], torch.ones_like(pred["g_pb"]))

        generator_loss = sum(_generator_loss.values())

        optimizer_G.zero_grad()
        generator_loss.backward()
        optimizer_G.step()

        pred["d_pp_fake"] = D(generated_img.detach(), condition_img)
        pred["d_pp_real"] = D(target_img, condition_img)

        _pp_discriminator_loss = {
            "real": gan_loss(pred["d_pp_real"], torch.ones_like(pred["d_pp_real"])),
            "fake": gan_loss(pred["d_pp_fake"], torch.zeros_like(pred["d_pp_fake"]))
        }

        pp_discriminator_loss = sum(_pp_discriminator_loss.values())/len(_pp_discriminator_loss)

        optimizer_D.zero_grad()
        pp_discriminator_loss.backward()
        optimizer_D.step()

        output = {
            "loss": {
                "g": {k: v.item() for k, v in _generator_loss.items()},
                "pp": {k: v.item() for k, v in _pp_discriminator_loss.items()},
                "g_total": generator_loss.item(),
                "pp_total": pp_discriminator_loss.item()
            },
        }

        if opt.use_db:
            pred["d_pb_fake"] = DB(fake_pb_pool.query(torch.cat([generated_img.detach(), target_pose], dim=1).data))
            pred["d_pb_real"] = DB(torch.cat([target_img, target_pose], dim=1))

            _pb_discriminator_loss = {
                "real": gan_loss(pred["d_pb_real"], torch.ones_like(pred["d_pb_real"])),
                "fake": gan_loss(pred["d_pb_fake"], torch.zeros_like(pred["d_pb_fake"]))
            }

            pb_discriminator_loss = sum(_pb_discriminator_loss.values()) / len(_pb_discriminator_loss)

            optimizer_DB.zero_grad()
            pb_discriminator_loss.backward()
            optimizer_DB.step()

            output["loss"]["pb"] = {k: v.item() for k, v in _pb_discriminator_loss.items()}
            output["loss"]["pb_total"] = pb_discriminator_loss.item()

        output["pred"] = {k: torch.sigmoid(v).mean().item() for k, v in pred.items()}

        if engine.state.iteration % opt.print_freq == 0:
            path = os.path.join(opt.output_dir, VAL_IMG_FNAME.format(engine.state.epoch, engine.state.iteration))
            get_current_visuals(path, batch, [generated_img])

        return output

    trainer = Engine(step)

    # attach running average metrics
    monitoring_metrics = ['loss_g_total', "loss_pp_total"]

    pred_names = ["g_pp", "d_pp_fake", "d_pp_real"]
    total_loss_names = ["g_total", "pp_total"]
    g_loss_names = ["adv", "mask_l1", "l1", "attr", "perceptual"]

    if opt.use_db:
        monitoring_metrics += ['loss_pb_total']
        pred_names += ["g_pb", "d_pb_fake", "d_pb_real"]
        total_loss_names += ["pb_total"]
        g_loss_names += ["adv_pb"]

    def make_ofn(keys):
        def ofn(x):
            for k in keys:
                x = x[k]
            return x
        return ofn

    running_averages = {"pred_{}".format(pn): make_ofn(["pred", pn]) for pn in pred_names}
    running_averages.update({"loss_g_{}".format(n): make_ofn(["loss", "g", n]) for n in g_loss_names})
    running_averages.update({"loss_{}".format(ln): make_ofn(["loss", ln]) for ln in total_loss_names})
    running_averages.update({"loss_pp_{}".format(n): make_ofn(["loss", "pp", n]) for n in ("real", "fake")})
    if opt.use_db:
        running_averages.update({"loss_pb_{}".format(n): make_ofn(["loss", "pb", n]) for n in ("real", "fake")})

    attach_engine(trainer, running_averages)

    networks_to_save = dict(G=G, D=D)

    def add_message(engine):
        message = ""
        for mm in monitoring_metrics + ['pred_g_pp', "pred_d_pp_fake"]:
            message += "<{}: {:.3f}> ".format(mm, engine.state.metrics[mm])
        return message

    warp_common_handler(
        trainer,
        opt,
        networks_to_save,
        monitoring_metrics,
        add_message,
        [FAKE_IMG_FNAME, VAL_IMG_FNAME]
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        print("-----------scheduler------step----------")
        print(optimizer_D.param_groups[0]["lr"])
        print(optimizer_G.param_groups[0]["lr"])
        scheduler_G.step()
        scheduler_D.step()
        if opt.use_db:
            scheduler_DB.step()
        print(optimizer_D.param_groups[0]["lr"])
        print(optimizer_G.param_groups[0]["lr"])
        print("-----------scheduler------over----------")

    val_data_pairs = _get_val_data_pairs(opt, device)
    @trainer.on(Events.ITERATION_COMPLETED)
    def save_example(engine):
        if engine.state.iteration > 0 and engine.state.iteration % opt.print_freq == 0:
            generated_img = G(val_data_pairs["P1"], torch.cat([val_data_pairs["BP1"], val_data_pairs["BP2"]], dim=1))
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
    parser.add_argument('--use_db', default=True, type=bool_arg)
    parser.add_argument('--flip_rate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0002)

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
