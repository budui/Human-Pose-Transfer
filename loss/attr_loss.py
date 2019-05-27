import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models


ATTR_NAMES = ["gender", "hair", "up", "down", "clothes", "hat", "backpack", "bag", "handbag", "age",
              "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen",
              "downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue",
              "downgreen", "downbrown"]
ATTR_NUM_CLASS = [2]*9+[4]+[2]*17


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True,
                 num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


class APR(nn.Module):
    def __init__(self, attributes, num_identity, drop_rate=0.5, last_stride=2):
        super().__init__()
        self.attributes = attributes
        base_model = models.resnet50(pretrained=True)
        if last_stride == 1:
            base_model.layer4[0].downsample[0].stride = (1, 1)
            base_model.layer4[0].conv2.stride = (1, 1)
        base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = base_model
        self.id_classifier = ClassBlock(2048, num_identity, drop_rate)
        self.attr_classifier = {}
        for attr in attributes:
            self.attr_classifier[attr] = nn.Linear(2048, attributes[attr])
            setattr(self, "attr_{}_classifier".format(attr), self.attr_classifier[attr])

    def forward(self, img):
        x = self.model.conv1(img)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        f = x.view(x.size(0), x.size(1))
        id_pred = self.id_classifier(f)
        attrs_pred = {}
        for attr in self.attributes:
            classifier = self.attr_classifier[attr]
            attrs_pred[attr] = classifier(f)
        return id_pred, attrs_pred


class IDAttrLoss(nn.Module):
    def __init__(self, apr_path, device="cuda"):
        super(IDAttrLoss, self).__init__()

        var_std = torch.Tensor([0.229, 0.224, 0.225]).resize_(1, 3, 1, 1).to(device)
        var_mean = torch.Tensor([0.485, 0.456, 0.406]).resize_(1, 3, 1, 1).to(device)
        self.re_norm = lambda image: (((image + 1) / 2) - var_mean) / var_std
        self.apr = APR(dict(zip(ATTR_NAMES, ATTR_NUM_CLASS)), 751)
        self.apr.load_state_dict(torch.load(apr_path))
        self.apr.eval()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, generated_image, attr_labels):
        # [-1, 1] to [0, 1] then Normalize
        generated_image_norm = self.re_norm(generated_image)
        _, attrs_pred = self.apr(generated_image_norm)
        attr_loss = {}
        for attr, pred in attrs_pred.items():
            attr_loss[attr] = self.loss_fn(pred, attr_labels[attr])
        return sum(attr_loss.values())/len(attr_loss)