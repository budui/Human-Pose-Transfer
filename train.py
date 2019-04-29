#!/usr/bin/env python3

import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset.dataset as dataset
import models.PG2 as PG2
import util.util as util
from loss.mask_l1 import MaskL1Loss
from util.image_pool import ImagePool


def get_current_visuals(img_path, data_pair, generated_img1, generated_img2=None):
    height, width, batch_size = data_pair["P1"].size(2), data_pair["P1"].size(3), data_pair["P1"].size(0)

    if generated_img2 is None:
        vis = np.zeros((height * batch_size, width * 6, 3)).astype(np.uint8)
    else:
        vis = np.zeros((height * batch_size, width * 7, 3)).astype(np.uint8)

    def make_vis(image_list, row_id):
        for img_id, img in enumerate(image_list):
            vis[height * row_id:height * (1 + row_id), width * img_id:width * (img_id + 1), :] = img

    for i in range(batch_size):
        input_P1 = util.tensor2im_(data_pair["P1"].data[i])
        input_P2 = util.tensor2im_(data_pair["P2"].data[i])
        input_p2_mask = util.tensor2im_(data_pair["MP2"].data[i])
        fake_p1 = util.tensor2im_(generated_img1.data[i])
        input_BP1 = util.draw_pose_from_map_(data_pair["BP1"].data[i])[0]
        input_BP2 = util.draw_pose_from_map_(data_pair["BP2"].data[i])[0]

        if generated_img2 is None:
            make_vis([input_P1, input_BP1, input_P2, input_BP2, input_p2_mask, fake_p1], i)
        else:
            fake_p2 = util.tensor2im_(generated_img2.data[i])
            make_vis([input_P1, input_BP1, input_P2, input_BP2, input_p2_mask, fake_p1, fake_p2], i)

    util.save_image(vis, img_path)


def move_data_pair_to(device, data_pair):
    # move data to GPU
    for k in data_pair:
        if "path" in k:
            # do not move string
            continue
        else:
            data_pair[k] = data_pair[k].to(device)


def generate_val_image(val_image_path, val_image_loader, device, model_g1, model_g2=None):
    data_pair = next(iter(val_image_loader))
    move_data_pair_to(device, data_pair)

    model_g1.eval()
    if model_g2 is None:
        img_g1 = model_g1(torch.cat([data_pair["P1"], data_pair["BP2"]], dim=1))
        get_current_visuals(val_image_path, data_pair, img_g1)
    else:
        model_g2.eval()
        img_g1 = model_g1(torch.cat([data_pair["P1"], data_pair["BP2"]], dim=1))
        diff_map = model_g2(torch.cat([img_g1, data_pair["P1"]], dim=1))
        img_g2 = diff_map + img_g1
        get_current_visuals(val_image_path, data_pair, img_g1, img_g2)
        model_g2.train()
    model_g1.eval()


def get_generated_img(generator_1, generator_2, condition_img, condition_pose):
    generator_1_img = generator_1(torch.cat([condition_img, condition_pose], dim=1))
    diff_img = generator_2(torch.cat([condition_img, generator_1_img], dim=1))
    return generator_1_img + diff_img



def run(opt, G1_model, G2_model, D_model,
        G1_optimizer, G2_optimizer, D_optimizer,
        G1_loss, G2_loss, D_loss,
        image_loader, val_image_loader, device, loss_lambda=10,
        run_stage_1=False
        ):
    G1_model.to(device)
    G2_model.to(device)
    D_model.to(device)

    if run_stage_1:
        print("--------STAGE 1----START---------")
        # stage 1
        # every epoch has `random_select_size`(default=4000) image_pairs
        # in paper, there are 22k*16 = 352k image_pairs
        # 352k/4000 = 88
        # so, we can set `stage_1_epoch_num`=88
        for epoch in range(1, opt.stage_1_epoch_num + 1):
            running_loss = 0.0
            since = time.time()
            for _, data_pair in enumerate(image_loader):
                move_data_pair_to(device, data_pair)
                img_from_g1 = G1_model(torch.cat([data_pair["P1"], data_pair["BP2"]], dim=1))
                G1_optimizer.zero_grad()
                loss_from_g1 = G1_loss(img_from_g1, data_pair["P2"], data_pair["MP2"])
                loss_from_g1.backward()
                G1_optimizer.step()

                # stat
                running_loss += loss_from_g1.item() * data_pair["P1"].size(0)

            time_elapsed = time.time() - since
            print("STEP:{} G1_Loss:{:.4f} {:.0f}s".format(epoch, running_loss / 4000, time_elapsed))

            if epoch % 1 == 0:
                generate_val_image("checkpoints/train_image/stage1_epoch_{}.jpg".format(epoch),
                                   val_image_loader, device, G1_model)
            if epoch % 5 == 0:
                torch.save(G1_model.state_dict(), "checkpoints/G1_stage1_{}.pth".format(epoch))

        torch.save(G1_model.state_dict(), "checkpoints/G1.pth")

        print("--------STAGE 1----END---------")
        print("save model G1")
    else:
        print("load G1 from checkpoints/G1.pth")
        G1_model.load_state_dict(torch.load("checkpoints/G1.pth"))

    # stage 2
    # every epoch has `random_select_size`(default=4000) image_pairs
    # in paper, there are 14k*16 = 224k image_pairs
    # 224k/4000 = 56
    # so, we can set `stage_1_epoch_num`=56
    print("--------STAGE 2----START---------")
    G2_MaskL1_loss, G2_BCE_loss = G2_loss
    for epoch in range(1, opt.stage_2_epoch_num + 1):
        running_loss_d = {
            "G2-BCE": 0.0,
            "G2-MaskL1": 0.0,
            "G2": 0.0,
            "D-Pos": 0.0,
            "D-Neg": 0.0,
            "D": 0.0
        }

        since = time.time()
        ## TODO 可能pool的size也影响结果
        fake_PP_pool = ImagePool(50)
        for step, data_pair in enumerate(image_loader):
            batch_size = data_pair["P1"].size(0)
            move_data_pair_to(device, data_pair)
            neg_label = torch.full((batch_size, 1), 0, device=device, requires_grad=False)
            pos_label = torch.full((batch_size, 1), 1, device=device, requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            G2_optimizer.zero_grad()
            # GAN loss
            generated_img = get_generated_img(G1_model, G2_model, data_pair["P1"], data_pair["BP2"])
            neg_output = D_model(torch.cat([data_pair["P1"], generated_img], dim=1))
            g2_bce_loss = G2_BCE_loss(neg_output, pos_label)
            # Generators loss
            g2_l1_loss = G2_MaskL1_loss(generated_img, data_pair["P2"], data_pair["MP2"])
            # total loss
            loss_from_g2 = g2_bce_loss + g2_l1_loss * loss_lambda

            loss_from_g2.backward()
            G2_optimizer.step()

            # ----------------------------------

            running_loss_d["G2-MaskL1"] += g2_l1_loss.item() * batch_size
            running_loss_d["G2-BCE"] += g2_bce_loss.item() * batch_size
            running_loss_d["G2"] += loss_from_g2.item() * batch_size

            # ---------------------
            #  Train Discriminator
            # ---------------------

            D_optimizer.zero_grad()

            # real loss
            pos_input = torch.cat([data_pair["P1"], data_pair["P2"]], dim=1)
            pos_output = D_model(pos_input)
            d_pos_loss = D_loss(pos_output, pos_label)

            # fake loss
            neg_input = torch.cat([data_pair["P1"], generated_img], dim=1)
            neg_output = D_model(neg_input.detach())
            d_neg_loss = D_loss(neg_output, neg_label)

            # total loss
            loss_from_D = (d_neg_loss + d_pos_loss) * 0.5

            loss_from_D.backward()
            D_optimizer.step()

            # ----------------------------------

            if step % 100 == 0:
                print("Epoch{}-{} D(neg)/D(pos): {:.4f}/{:.4f}".format(
                    epoch, step,
                    neg_output.mean().item(),
                    pos_output.mean().item(),
                )
                )
            running_loss_d["D-Pos"] += d_pos_loss.item() * batch_size
            running_loss_d["D-Neg"] += d_neg_loss.item() * batch_size
            running_loss_d["D"] += loss_from_D.item() * batch_size

        print(
            "Epoch:{} G2_Loss(All/L1/BCE): {:.4f}/{:.4f}/{:.4f} D_Loss(All/Neg/Pos): {:.4f}/{:.4f}/{:.4f} {:.0f}s\n".format(
                epoch,
                running_loss_d["G2"]/4000, running_loss_d["G2-MaskL1"]/4000, running_loss_d["G2-BCE"]/4000,
                running_loss_d["D"]/4000, running_loss_d["D-Neg"]/4000, running_loss_d["D-Pos"]/4000,
                time.time() - since)
        )
        if epoch % 1 == 0:
            generate_val_image("checkpoints/train_image/stage2_epoch_{:0>3d}.jpg".format(epoch),
                               val_image_loader, device, G1_model, G2_model)
        if epoch % 5 == 0:
            torch.save(G2_model.state_dict(), "checkpoints/G2_stage2_{}.pth".format(epoch))
            torch.save(D_model.state_dict(), "checkpoints/D_stage2_{}.pth".format(epoch))

    print("--------STAGE 2----END---------")
    torch.save(G2_model.state_dict(), "checkpoints/G2.pth")
    print("save model G2")
    torch.save(D_model.state_dict(), "checkpoints/D.pth")
    print("save model D")


def main():
    parser = ArgumentParser(description='Training')
    parser.add_argument('--gpu_id', default=2, type=int, help='gpu_id: e.g. 0')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument("--stage_1_epoch_num", default=100, type=int, help="stage_1_epoch_num")
    parser.add_argument("--stage_2_epoch_num", default=80, type=int, help="stage_2_epoch_num")
    parser.add_argument('--d_lr', type=float, default=0.00002)
    parser.add_argument('--g_lr', type=float, default=0.00002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    cudnn.benchmark = True
    device = 'cuda'

    G1_model = PG2.G1(3 + 18, repeat_num=5, half_width=True, middle_z_dim=64)
    G2_model = PG2.G2(3 + 3, hidden_num=64, repeat_num=4, skip_connect=1)
    D_model = PG2.Discriminator(in_channels=6)

    gen_train_op1 = optim.Adam(G1_model.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2))
    gen_train_op2 = optim.Adam(G2_model.parameters(), lr=opt.g_lr, betas=(opt.beta1, opt.beta2))
    dis_train_op1 = optim.Adam(D_model.parameters(), lr=opt.d_lr, betas=(opt.beta1, opt.beta2))

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

    if not os.path.exists("checkpoints/train_image/"):
        os.makedirs("checkpoints/train_image/")

    run(opt, G1_model, G2_model, D_model, gen_train_op1, gen_train_op2, dis_train_op1,
        MaskL1Loss().cuda(), (MaskL1Loss().cuda(), nn.BCELoss().cuda()), nn.BCELoss().cuda(),
        image_loader, val_image_loader, device)


if __name__ == '__main__':
    main()
