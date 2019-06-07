from models import PAGAN
import torch
from train.helper import move_data_pair_to

from util.util import diagnose_network

def get_generator(path, device="cuda"):
    G = PAGAN.PAGenerator()
    diagnose_network(G, "G")
    G.eval()
    G.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    G.to(device)

    def generator(batch):
        move_data_pair_to(device, batch)
        condition_img = batch["P1"]
        generated_imgs = G(condition_img, torch.cat([batch["BP1"], batch["BP2"]], dim=1))
        generated_imgs.clamp_(-1, 1)
        return generated_imgs

    return generator
