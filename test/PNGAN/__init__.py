import torch

from models.PNGAN import ResGenerator
from train.helper import move_data_pair_to


def get_generator(path, num_res, device="cuda", generate_all=False):
    G = ResGenerator(64, num_res)
    G.eval()
    G.load_state_dict(torch.load(path))
    G.to(device)

    def generator(batch):
        move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        target_pose = batch["BP2"]
        if generate_all:
            g_imgs = []
            for i in range(num_res):
                g_imgs.append(G(condition_img, torch.cat([batch["BP1"], batch["BP2"]], dim=1), i))
            generated_imgs = torch.cat(g_imgs, dim=-1)
        else:
            generated_imgs = G(condition_img, torch.cat([batch["BP1"], batch["BP2"]], dim=1))
        generated_imgs.clamp_(-1, 1)
        return generated_imgs

    return generator
