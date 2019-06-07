import torch

from models import PG2
from test.helper import move_data_pair_to

from util.util import diagnose_network

def get_generator(G1_path, G2_path, device):
    generator_1 = PG2.G1(3 + 18, repeat_num=5, half_width=True, middle_z_dim=64)
    generator_1.load_state_dict(torch.load(G1_path, map_location="cpu"))
    diagnose_network(generator_1, "generator1")
    generator_2 = PG2.G2(3 + 3, hidden_num=64, repeat_num=3, skip_connect=1)
    diagnose_network(generator_2, "generator2")
    generator_2.load_state_dict(torch.load(G2_path, map_location="cpu"))
    generator_1.to(device)
    generator_2.to(device)

    def generator(batch):
        move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        condition_pose = batch["BP2"]

        # get generated img
        generator_1_imgs = generator_1(torch.cat([condition_img, condition_pose], dim=1))
        diff_imgs = generator_2(torch.cat([condition_img, generator_1_imgs], dim=1))
        generated_imgs = generator_1_imgs + diff_imgs
        generated_imgs.clamp_(-1, 1)
        return generated_imgs

    return generator
