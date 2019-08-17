import torch

from .model import Generator2, Generator1


def make_generator(config, device=torch.device("cuda")):
    cfg = config["model"]["generator1"]
    generator1 = Generator1(3 + 18, cfg["num_repeat"], cfg["middle_features_dim"], cfg["channels_base"],
                            cfg["image_size"])
    generator1.to(device)
    generator1.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

    cfg = config["model"]["generator2"]
    generator2 = Generator2(3 + 3, cfg["channels_base"], cfg["num_repeat"], cfg["num_skip_out_connect"])
    generator2.to(device)
    generator2.load_state_dict(torch.load(cfg["pretrained_path"], map_location="cpu"))

    def generate(batch):
        with torch.no_grad():
            generator1.eval()
            generator2.eval()
            generated_img_1 = generator1(batch["condition_img"], batch["target_bone"])
            generated_img = generator2(batch["condition_img"], generated_img_1) + generated_img_1
            return generated_img

    return generate
