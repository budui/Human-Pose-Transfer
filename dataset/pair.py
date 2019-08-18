import csv
import os

from .base import BoneDataset, wrap_dict_name


class PairBoneDataset(BoneDataset):
    @staticmethod
    def load_pair_list(pair_list_path):
        assert os.path.isfile(pair_list_path)
        with open(pair_list_path, "r") as f:
            f_csv = csv.reader(f)
            next(f_csv)
            pair_list = [tuple(item) for item in f_csv]
            return pair_list

    def __init__(self, pair_list_path, *nargs, **kwargs):
        super().__init__(*nargs, **kwargs)
        self.pair_list_path = pair_list_path
        self.pairs = self.load_pair_list(pair_list_path)

    def __getitem__(self, input_idx):
        img_p1_name, img_p2_name = self.pairs[input_idx]

        pair = wrap_dict_name(self.prepare_item(img_p1_name), "condition_")
        pair.update(wrap_dict_name(self.prepare_item(img_p2_name), "target_"))

        return pair

    def __len__(self):
        return len(self.pairs)

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
