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
                 flip_rate=0.0, loader=default_loader,
                 transform=DEFAULT_TRANS, only_path=False):

        self.image_folder = image_folder
        self.bone_folder = bone_folder
        self.mask_folder = mask_folder

        self.flip_rate = flip_rate
        self.use_flip = self.flip_rate > 0.0

        self.size, self.pairs = self.load_pair_list(pair_list_path)
        self.key_points = self.load_key_points(annotations_file_path)

        self.transform = transform
        self.loader = loader

        self.only_path = only_path

    def __repr__(self):
        return "<BoneDataset size: {} flip_rate: {}>".format(
            len(self), self.flip_rate
        )

    @staticmethod
    def load_pair_list(pair_list_path):
        assert os.path.isfile(pair_list_path)
        with open(pair_list_path, "r") as f:
            f_csv = csv.reader(f)
            next(f_csv)
            pair_list = [tuple(item) for item in f_csv]
            return len(pair_list), pair_list

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

    def __getitem__(self, input_idx):
        img_p1_name, img_p2_name = self.pairs[input_idx]

        if self.use_flip:
            flip = torch.rand(1).item() < self.flip_rate
        else:
            flip = False

        if self.only_path:
            return {'P1_path': img_p1_name, 'P2_path': img_p2_name}

        img_p1 = self.load_image_data(img_p1_name, flip)
        img_p2 = self.load_image_data(img_p2_name, flip)
        bone_p1 = self.load_bone_data(img_p1_name, flip)
        bone_p2 = self.load_bone_data(img_p2_name, flip)
        mask_p1 = self.load_mask_data(img_p1_name, flip)
        mask_p2 = self.load_mask_data(img_p2_name, flip)

        return {'P1': img_p1, 'BP1': bone_p1, 'P1_path': img_p1_name,
                'MP1': mask_p1, 'KP1': self.key_points[img_p1_name],
                'P2': img_p2, 'BP2': bone_p2, 'P2_path': img_p2_name,
                'MP2': mask_p2, 'KP2': self.key_points[img_p2_name]}

    def __len__(self):
        return self.size


class AttrBoneDataset(BoneDataset):
    def __init__(self, mat_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = mat_path
        self.mat = loadmat(self.path)["market_attribute"][0][0]
        self.attrs = self._make_attr_dict("test")
        self.attrs.update(self._make_attr_dict("train"))

    def __getitem__(self, idx):
        basic_items = super().__getitem__(idx)
        id1 = basic_items["P1_path"][:4]
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


def _test():
    image_dataset = BoneDataset(
        os.path.join("", "bounding_box_train/"),
        "data/market/train/pose_map_image/",
        "data/market/train/pose_mask_image/",
        "data/market/pairs-train.csv",
        "data/market/annotation-train.csv",
        flip_rate=0.5,
        only_path=True
    )

    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32,
                                               num_workers=8, pin_memory=True,
                                               sampler=torch.utils.data.RandomSampler(image_dataset, replacement=True,
                                                                                      num_samples=4000)
                                               )
    print(len(image_loader), image_dataset)
    iter_results = set()
    for j in range(120):
        for data in image_loader:
            for i in range(len(data["P1_path"])):
                iter_results.add((data["P1_path"][i], data["P2_path"][i]))
        print(":", len(iter_results))


if __name__ == '__main__':
    _test()
