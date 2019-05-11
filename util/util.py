from PIL import Image
import numpy as np

from util import pose


def tensor2image(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def get_current_visuals(img_path, data_pair, new_imgs=None):
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
                new_img_list.append(tensor2image(nimg.data[i]))
        input_P1 = tensor2image(data_pair["P1"].data[i])
        image_size = input_P1.shape[:2]
        input_P2 = tensor2image(data_pair["P2"].data[i])
        input_p2_mask = tensor2image(data_pair["MP2"].data[i])
        input_BP1 = pose.draw_pose_from_cords(data_pair["KP1"].data[i], image_size)[0]
        input_BP2 = pose.draw_pose_from_cords(data_pair["KP2"].data[i], image_size)[0]

        make_vis([input_P1, input_BP1, input_P2, input_BP2, input_p2_mask] + new_img_list, i)

    save_image(vis, img_path)