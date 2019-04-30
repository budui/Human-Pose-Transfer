import torch


def interpolation(pose_1, pose_2, pose_encoder, pose_decoder, num_inner_points=5):
    z1 = pose_encoder(pose_1)
    z2 = pose_encoder(pose_2)
    after_interpolation = [i*(z2-z1)/(num_inner_points+1) + z1 for i in range(num_inner_points+2)]
    after_interpolation_batch = torch.cat([z.unsqueeze(0) for z in after_interpolation], dim=0)
    pose_interpolation = pose_decoder(after_interpolation_batch)
    return pose_interpolation


def _test():
    from models.DPIG import PoseDecoder, PoseEncoder
    from test.PoseSample import _load_model
    from dataset.key_point_dataset import KeyPointDataset
    from util.vis.pose import show as show_pose

    key_points_dir = "data/market/annotation-test.csv"
    dataset = KeyPointDataset(key_points_dir)

    device = "cuda"

    pose_1 = dataset[1]
    pose_2 = dataset[3]
    pose_1 = pose_1.to(device)
    pose_2 = pose_2.to(device)


    encoder_path = "./ckp/networks_pose_encoder_12.pth"
    decoder_path = "./ckp/networks_pose_decoder_12.pth"

    pose_encoder = _load_model(PoseEncoder, encoder_path, device)
    pose_decoder = _load_model(PoseDecoder, decoder_path, device)

    pose_interpolation = interpolation(pose_1, pose_2, pose_encoder, pose_decoder)
    new_poses = [pose.squeeze(0) for pose in pose_interpolation]
    show_pose(new_poses, "new_pose.jpg")


if __name__ == '__main__':
    _test()