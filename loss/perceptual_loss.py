import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self, perceptual_layers=3, device="cuda"):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.vgg_sub_model = nn.Sequential()
        for i, layer in enumerate(list(vgg)):
            self.vgg_sub_model.add_module(str(i), layer)
            if i == perceptual_layers:
                break
        self.vgg_sub_model.to(device)
        self.var_std = torch.Tensor([0.229, 0.224, 0.225]).resize_(1, 3, 1, 1).to(device)
        self.var_mean = torch.Tensor([0.485, 0.456, 0.406]).resize_(1, 3, 1, 1).to(device)

    def forward(self, generated_image, origin_image):
        # [-1, 1] to [0, 1] then Normalize
        re_norm = lambda image: (((image + 1) / 2) - self.var_mean) / self.var_std
        generated_image_norm = re_norm(generated_image)
        origin_image_norm = re_norm(origin_image)

        g_feature = self.vgg_sub_model(generated_image_norm)
        o_feature = self.vgg_sub_model(origin_image_norm)
        o_feature = o_feature.detach()

        loss = F.l1_loss(g_feature, o_feature)
        return loss


def _test():
    p = PerceptualLoss()
    print(p)


if __name__ == '__main__':
    _test()
