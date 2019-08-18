import csv
import json
import os

import numpy
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

DEFAULT_TRANS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def wrap_dict_name(d, prefix):
    return {
        "{}{}".format(prefix, key): value
        for key, value in d.items()
    }


class BoneDataset(Dataset):
    def __init__(self, image_folder, bone_folder, mask_folder, annotations_file_path,
                 exclude_fields=None, flip_rate=0.0, loader=default_loader, transform=DEFAULT_TRANS):
        self.image_folder = image_folder
        self.bone_folder = bone_folder
        self.mask_folder = mask_folder

        self.flip_rate = flip_rate
        self.use_flip = self.flip_rate > 0.0

        self.exclude_fields = [] if exclude_fields is None else exclude_fields

        self.key_points = self.load_key_points(annotations_file_path)

        self.transform = transform
        self.loader = loader

    @staticmethod
    def load_key_points(annotations_file_path):
        with open(annotations_file_path, "r") as f:
            f_csv = csv.reader(f, delimiter=":")
            next(f_csv)
            annotations_data = {}
            for row in f_csv:
                img_name = row[0]
                key_points_y = json.loads(row[1])
                key_points_x = json.loads(row[2])
                annotations_data[img_name] = torch.cat([
                    torch.tensor(key_points_y).unsqueeze_(-1),
                    torch.tensor(key_points_x).unsqueeze_(-1)
                ], dim=-1)
            return annotations_data

    @staticmethod
    def load_bone_data(bone_folder, img_name, flip=False):
        bone_img = numpy.load(os.path.join(bone_folder, img_name + ".npy"))
        bone = torch.from_numpy(bone_img).float()  # h, w, c
        bone = bone.transpose(2, 0)  # c,w,h
        bone = bone.transpose(2, 1)  # c,h,w
        if flip:
            bone = bone.flip(dims=[-1])
        return bone

    @staticmethod
    def load_mask_data(mask_folder, img_name, flip=False):
        mask = torch.as_tensor(numpy.load(os.path.join(mask_folder, img_name + ".npy")), dtype=torch.float)
        if flip:
            mask = mask.flip(dims=[-1])
        mask = mask.unsqueeze(0).expand(3, -1, -1)
        return mask

    def load_image_data(self, path, flip=False):
        try:
            img = self.loader(os.path.join(self.image_folder, path))
        except FileNotFoundError as e:
            print(path)
            raise e

        if self.transform is not None:
            img = self.transform(img)
        if flip:
            img = img.flip(dims=[-1])
        return img

    def prepare_item(self, image_name):
        flip = torch.rand(1).item() < self.flip_rate if self.use_flip else False

        item = {"path": image_name}
        if "img" not in self.exclude_fields:
            item["img"] = self.load_image_data(image_name, flip)
        if "bone" not in self.exclude_fields:
            item["bone"] = self.load_bone_data(self.bone_folder, image_name, flip)
        if "mask" not in self.exclude_fields:
            item["mask"] = self.load_mask_data(self.mask_folder, image_name, flip)
        if "key_points" not in self.exclude_fields:
            item["key_points"] = self.key_points[image_name]
        return item

    def __getitem__(self, input_idx):
        pass

    def __len__(self):
        pass
