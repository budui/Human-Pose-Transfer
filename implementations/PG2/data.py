from torch.utils.data import DataLoader, RandomSampler

import dataset


def get_data_loader(config):
    cfg = config["dataset"]["path"]["train"]
    image_dataset = dataset.BoneDataset(cfg["image"], cfg["bone"], cfg["mask"], cfg["pair"], cfg["annotation"],
                                        flip_rate=config["train"]["data"]["flip_rate"])
    image_loader = DataLoader(image_dataset, batch_size=config["train"]["batch_size"],
                              num_workers=8, pin_memory=True, drop_last=True,
                              sampler=RandomSampler(image_dataset, replacement=config["train"]["data"]["replacement"]))
    print(image_dataset)
    return image_loader


def get_val_data_pairs(config):
    cfg = config["dataset"]["path"]["test"]
    image_dataset = dataset.BoneDataset(cfg["image"], cfg["bone"], cfg["mask"], cfg["pair"], cfg["annotation"])
    dl = DataLoader(
        image_dataset, num_workers=1,
        batch_size=config["log"]["verify"]["batch_size"],
        shuffle=config["log"]["verify"]["shuffle"],
    )
    return next(iter(dl))
