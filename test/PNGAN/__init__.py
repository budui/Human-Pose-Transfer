import torch

from models.PNGAN import ResGenerator
from train.helper import move_data_pair_to


def get_generator(path, num_res, device="cuda"):
    G = ResGenerator(64, num_res)
    G.load_state_dict(torch.load(path))
    G.to(device)

    def generator(batch):
        move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        target_pose = batch["BP2"]

        generated_imgs = G(condition_img, target_pose)
        generated_imgs.clamp_(-1, 1)
        return generated_imgs
    return generator
