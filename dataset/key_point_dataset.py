import csv
import json
import os

import torch
from torch.utils.data import Dataset

KEY_POINTS_NUM = 18
INVALID_KEY_POINTS_VALUE = -1
IMG_SIZE = (128, 64)


class KeyPointDataset(Dataset):
    def __init__(self, annotation_file_path, norm=True):
        assert os.path.exists(annotation_file_path)
        self.annotation_file_path = annotation_file_path
        self.norm = norm
        self.coordinates, self.names = self._load_annotations(annotation_file_path)
        self.visibility = self._check_visibility(self.coordinates)
        self.dicts = dict(zip(self.names, range(len(self.coordinates))))

    @staticmethod
    def _load_annotations(annotations_file_path):
        with open(annotations_file_path, "r") as f:
            f_csv = csv.reader(f, delimiter=":")
            next(f_csv)
            annotations_data = []
            names = []
            for row in f_csv:
                img_name = row[0]
                names.append(img_name)
                key_points_y = json.loads(row[1])
                key_points_x = json.loads(row[2])
                annotations_data.append(torch.cat([
                    torch.Tensor(key_points_y).unsqueeze_(-1),
                    torch.Tensor(key_points_x).unsqueeze_(-1)
                ], dim=-1))
            return annotations_data, names

    @staticmethod
    def _check_visibility(coordinates):
        visibility = []
        for kps in coordinates:
            visibility.append(torch.Tensor(
                [0 if INVALID_KEY_POINTS_VALUE in kp else 1 for kp in kps]
            ).unsqueeze_(-1))
        return visibility

    def __len__(self):
        return len(self.visibility)

    def __getitem__(self, index):
        if not self.norm:
            return torch.cat([self.coordinates[index], self.visibility[index]], dim=-1)
        else:
            norm_coord = self.coordinates[index].div(torch.Tensor([IMG_SIZE]).expand([KEY_POINTS_NUM, 2]))
            return torch.cat([norm_coord, self.visibility[index]], dim=-1)

    def get(self, name1, name2):
        index1 = self.dicts[name1]
        index2 = self.dicts[name2]
        return self.__getitem__(index1), self.__getitem__(index2)

    def __repr__(self):
        return "<KeyPointDataset path:{path} number:{len} norm:{norm})>".format(
            path=self.annotation_file_path, len=len(self), norm=self.norm
        )


if __name__ == '__main__':
    kpd = KeyPointDataset("./data/market/annotation-train.csv", True)
    print(kpd[100])
    print(kpd[101].size())
