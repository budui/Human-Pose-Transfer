import csv
import json
import os

import numpy
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader

DEFAULT_TRANS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

ATTR_NAMES = ["gender", "hair", "up", "down", "clothes", "hat", "backpack", "bag", "handbag", "age",
              "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen",
              "downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue",
              "downgreen", "downbrown"]

ATTR_NUM_CLASS = [2] * 9 + [4] + [2] * 17


class BoneDataset(Dataset):
    def __init__(self, image_folder, bone_folder, mask_folder, pair_list_path, annotations_file_path,
                 flip_rate=0.0, loader=default_loader, transform=DEFAULT_TRANS):

        self.image_folder = image_folder
        self.bone_folder = bone_folder
        self.mask_folder = mask_folder
        self.pair_list_path = pair_list_path

        self.flip_rate = flip_rate
        self.use_flip = self.flip_rate > 0.0

        self.pairs = self.load_pair_list(pair_list_path)
        self.key_points = self.load_key_points(annotations_file_path)

        self.transform = transform
        self.loader = loader

    def __repr__(self):
        return """
{}(
    size: {},
    flip_rate: {},
    image_folder: {},
    bone_folder: {},
    mask_folder: {}
    pair_list_path: {}
)
transform: {}
""".format(self.__class__, len(self), self.flip_rate, self.image_folder,
           self.bone_folder, self.mask_folder, self.pair_list_path, self.transform)

    @staticmethod
    def load_pair_list(pair_list_path):
        assert os.path.isfile(pair_list_path)
        with open(pair_list_path, "r") as f:
            f_csv = csv.reader(f)
            next(f_csv)
            pair_list = [tuple(item) for item in f_csv]
            return pair_list

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
                    torch.Tensor(key_points_y).unsqueeze_(-1),
                    torch.Tensor(key_points_x).unsqueeze_(-1)
                ], dim=-1)
            return annotations_data

    def load_bone_data(self, img_name, flip=False):
        bone_img = numpy.load(os.path.join(self.bone_folder, img_name + ".npy"))
        bone = torch.from_numpy(bone_img).float()  # h, w, c
        bone = bone.transpose(2, 0)  # c,w,h
        bone = bone.transpose(2, 1)  # c,h,w
        if flip:
            bone = bone.flip(dims=[-1])
        return bone

    def load_mask_data(self, img_name, flip=False):
        mask = torch.Tensor(numpy.load(os.path.join(self.mask_folder, img_name + ".npy")).astype(int))
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
        img = self.load_image_data(image_name, flip)
        bone = self.load_bone_data(image_name, flip)
        mask = self.load_mask_data(image_name, flip)
        key_points = self.key_points[image_name]
        return img, bone, mask, key_points

    def __getitem__(self, input_idx):
        img_p1_name, img_p2_name = self.pairs[input_idx]

        output_item_fields = ["path", "img", "bone", "mask", "key_points"]
        output = dict(zip(
            ["condition_{}".format(field) for field in output_item_fields],
            [img_p1_name, *self.prepare_item(img_p1_name)]
        ))
        output.update(dict(zip(
            ["target_{}".format(field) for field in output_item_fields],
            [img_p2_name, *self.prepare_item(img_p2_name)]
        )))

        return output

    def __len__(self):
        return len(self.pairs)


class AttrBoneDataset(BoneDataset):
    def __init__(self, mat_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = mat_path
        self.mat = loadmat(self.path)["market_attribute"][0][0]
        self.attrs = self._make_attr_dict("test")
        self.attrs.update(self._make_attr_dict("train"))

    def __getitem__(self, idx):
        basic_items = super().__getitem__(idx)
        id1 = basic_items["c_path"][:4]
        attr_id1 = self.attrs[id1]

        basic_items.update({"attr": attr_id1})
        return basic_items

    def __repr__(self):
        return "<AttrBoneDataset size: {} flip_rate: {} attribute_path: {}>".format(
            len(self), self.flip_rate, self.path
        )

    def _make_attr_dict(self, t="train"):
        mat = self.mat[t][0][0]
        identities = mat["image_index"][0]
        attrs = {}
        for an in ATTR_NAMES:
            attrs[an] = mat[an][0]

        attrs_per_id = {}
        for i, ids in enumerate(identities):
            attr_person = dict()
            for num_c, an in zip(ATTR_NUM_CLASS, ATTR_NAMES):
                attr_person[an] = int(attrs[an][i] - 1)
                if attr_person[an] >= num_c or attr_person[an] < 0:
                    raise ValueError("num classes")
            attrs_per_id[ids[0]] = attr_person
        return attrs_per_id
