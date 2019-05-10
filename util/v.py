from . import util
import numpy as np


def get_current_visuals_(img_path, data_pair, new_imgs=None):
    height, width, batch_size = data_pair["P1"].size(2), data_pair["P1"].size(3), data_pair["P1"].size(0)

    image_num = batch_size if batch_size < 6 else 6

    vis = np.zeros((height * image_num, width * (5 + len(new_imgs)), 3)).astype(np.uint8)

    def make_vis(image_list, row_id):
        for img_id, img in enumerate(image_list):
            vis[height * row_id:height * (1 + row_id), width * img_id:width * (img_id + 1), :] = img

    for i in range(image_num):
        new_img_list = []
        if new_imgs is not None:
            for nimg in new_imgs:
                new_img_list.append(util.tensor2image(nimg.data[i]))
        input_P1 = util.tensor2image(data_pair["P1"].data[i])
        input_P2 = util.tensor2image(data_pair["P2"].data[i])
        input_p2_mask = util.tensor2image(data_pair["MP2"].data[i])
        input_BP1 = util.draw_pose_from_map(data_pair["BP1"].data[i])[0]
        input_BP2 = util.draw_pose_from_map(data_pair["BP2"].data[i])[0]

        make_vis([input_P1, input_BP1, input_P2, input_BP2, input_p2_mask] + new_img_list, i)

    util.save_image(vis, img_path)


def get_current_visuals(img_path, data_pair, generated_img1, generated_img2=None):
    height, width, batch_size = data_pair["P1"].size(2), data_pair["P1"].size(3), data_pair["P1"].size(0)

    image_num = batch_size if batch_size < 6 else 6

    if generated_img2 is None:
        vis = np.zeros((height * image_num, width * 6, 3)).astype(np.uint8)
    else:
        vis = np.zeros((height * image_num, width * 7, 3)).astype(np.uint8)

    def make_vis(image_list, row_id):
        for img_id, img in enumerate(image_list):
            vis[height * row_id:height * (1 + row_id), width * img_id:width * (img_id + 1), :] = img

    for i in range(image_num):
        input_P1 = util.tensor2image(data_pair["P1"].data[i])
        input_P2 = util.tensor2image(data_pair["P2"].data[i])
        input_p2_mask = util.tensor2image(data_pair["MP2"].data[i])
        fake_p1 = util.tensor2image(generated_img1.data[i])
        input_BP1 = util.draw_pose_from_map(data_pair["BP1"].data[i])[0]
        input_BP2 = util.draw_pose_from_map(data_pair["BP2"].data[i])[0]

        if generated_img2 is None:
            make_vis([input_P1, input_BP1, input_P2, input_BP2, input_p2_mask, fake_p1], i)
        else:
            fake_p2 = util.tensor2image(generated_img2.data[i])
            make_vis([input_P1, input_BP1, input_P2, input_BP2, input_p2_mask, fake_p1, fake_p2], i)

    util.save_image(vis, img_path)