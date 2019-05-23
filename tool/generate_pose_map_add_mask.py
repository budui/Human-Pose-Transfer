#!/usr/bin/env python3

import csv
import json
import os
from argparse import ArgumentParser

import numpy as np
from skimage.draw import circle
from skimage.morphology import square, dilation, erosion

# default value when HPM detect failed
KEY_POINT_MISSING_VALUE = -1
# key points connection: list of (from_point_index, to_point_index)
LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
            [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
            [0, 15], [15, 17], [2, 16], [5, 17]]


def load_annotations_from_file(annotations_file_path):
    with open(annotations_file_path, "r") as f:
        f_csv = csv.reader(f, delimiter=":")
        next(f_csv)
        annotations_data = []
        for row in f_csv:
            annotations_data.append((row[0], np.concatenate([np.expand_dims(json.loads(row[1]), -1),
                                                             np.expand_dims(json.loads(row[2]), -1)], axis=1)))
        return annotations_data


def expand_key_points(key_points, radius=4):
    new_points = []
    for f, t in LIMB_SEQ:
        if KEY_POINT_MISSING_VALUE in key_points[f] or KEY_POINT_MISSING_VALUE in key_points[t]:
            continue
        from_point, to_point = key_points[f], key_points[t]
        pair_distance = np.linalg.norm(from_point - to_point)
        new_points_num = int(pair_distance / radius)
        for i in range(1, new_points_num):
            new_points.append((from_point + (i / new_points_num) * (to_point - from_point)))
    return new_points


def key_point_to_mask(key_points, img_size, radius=6):
    new_points = expand_key_points(key_points, radius)
    mask = np.zeros(shape=img_size, dtype=bool)

    for i, joint in enumerate(list(key_points) + new_points):
        if KEY_POINT_MISSING_VALUE in joint:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        mask[yy, xx] = True
    mask = dilation(mask, square(radius + 3))
    mask = erosion(mask, square(radius + 3))
    return mask


def key_point_to_map(key_points, img_size, sigma=6):
    map_image = np.zeros(img_size + key_points.shape[0:1], dtype="float32")
    for i, point in enumerate(key_points):
        if KEY_POINT_MISSING_VALUE in point:
            # useless point
            continue
        # get image coordinate.
        # if image_size = (4, 2)
        # image:
        # _________________
        # | (0, 0) (1, 0) |
        # | (0, 1) (1, 1) |
        # | (0, 2) (1, 2) |
        # | (0, 3) (1, 3) |
        # _________________
        # xx = array([[0, 1],
        #        [0, 1],
        #        [0, 1],
        #        [0, 1]])
        # yy = array([[0, 0],
        #        [1, 1],
        #        [2, 2],
        #        [3, 3]])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        # @latex: e^{-\frac{D_x^2+D_y^2}{2\sigma^2}}
        map_image[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return map_image


def compute_pose(annotations_file_path, map_save_path, mask_save_path, image_size):
    annotations_data = load_annotations_from_file(annotations_file_path)
    annotations_count = len(annotations_data)
    for i, item in enumerate(annotations_data):
        if i % 100 == 0:
            print("processing {0}/{1}".format(i, annotations_count))
        img_name, key_points = item

        pose_map = key_point_to_map(key_points, image_size)
        np.save(os.path.join(map_save_path, img_name + ".npy"), pose_map)
        pose_mask = key_point_to_mask(key_points, image_size)
        np.save(os.path.join(mask_save_path, img_name + ".npy"), pose_mask)


def main(dataset, d_type):
    annotations_file_path = "data/{dataset}/annotation-{type}.csv".format(dataset=dataset, type=d_type)
    pose_map_save_path = "data/{dataset}/{type}/pose_map_image/".format(dataset=dataset, type=d_type)
    pose_mask_save_path = "data/{dataset}/{type}/pose_mask_image/".format(dataset=dataset, type=d_type)

    os.makedirs(pose_mask_save_path, exist_ok=True)
    os.makedirs(pose_map_save_path, exist_ok=True)
    compute_pose(annotations_file_path, pose_map_save_path, pose_mask_save_path, (128, 64))


if __name__ == '__main__':
    parser = ArgumentParser(description='generate 18-channels pose image and use pose to draw mask')
    parser.add_argument("--dataset", type=str, default="market")
    parser.add_argument("--type", type=str, default="train")
    opt = parser.parse_args()
    main(opt.dataset, opt.type)
