import os
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.io import loadmat
from torch.nn import init
from torchvision import models
from torchvision import transforms
import torch.backends.cudnn as cudnn
import sys
from torchvision.datasets.folder import default_loader

ATTR_NAMES = ["gender", "hair", "up", "down", "clothes", "hat", "backpack", "bag", "handbag", "age",
              "upblack", "upwhite", "upred", "uppurple", "upyellow", "upgray", "upblue", "upgreen",
              "downblack", "downwhite", "downpink", "downpurple", "downyellow", "downgray", "downblue",
              "downgreen", "downbrown"]

ATTR_NUM_CLASS = [2] * 9 + [4] + [2] * 17


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
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
        if droprate > 0:
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
            return x, f
        else:
            x = self.classifier(x)
            return x


class APR(nn.Module):
    def __init__(self, attributes, num_identity, drop_rate=0.5, last_stride=2):
        super().__init__()
        self.attributes = attributes
        base_model = models.resnet50(pretrained=False)
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


class PDataset(data.Dataset):
    def __init__(self, path, mat_path="data/market/attribute/market_attribute.mat", transfrom=None, loader=default_loader):
        assert os.path.exists(path) and os.path.exists(mat_path)
        self.path = path
        self.mat = loadmat(mat_path)["market_attribute"][0][0]
        self.attrs = self._make_attr_dict("test")
        self.transform = transfrom
        self.loader = loader

        file_list = os.listdir(self.path)
        self.file_list = []
        for item in  file_list:
            if not item.endswith('.jpg') and not item.endswith('.png'):
                continue
            self.file_list.append(item)

        with open("market_name_to_id_test.json", "r") as f:
            import json
            data = json.load(f)
        self.name_to_id = data

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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, pid):
        img_path = self.file_list[pid]
        id_name = img_path[:4]
        attr = self.attrs[id_name]
        class_id = self.name_to_id[id_name]
        image = self.loader(os.path.join(self.path, img_path))
        if self.transform is not None:
            image = self.transform(image)
            image = image[:, :, 4 * 64:5 * 64]
            #image = image[:, :,  2 * 64:3 * 64]
        return image, class_id, attr


torch.cuda.set_device(0)
cudnn.benchmark = True

apr = APR(dict(zip(ATTR_NAMES, ATTR_NUM_CLASS)), 751)

apr.load_state_dict(torch.load("data/net_test_ARP.pth"))
apr.to("cuda")
apr.eval()

device = "cuda"

pd = PDataset(sys.argv[1], transfrom=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]))
dl = torch.utils.data.DataLoader(pd, batch_size=32, num_workers=8, drop_last=False)
loss_fn = nn.CrossEntropyLoss()

loss_t = 0

for batch in dl:
    img, pid, attr_labels = batch
    img = img.to(device)
    pid = pid.to(device)
    for k in attr_labels:
        attr_labels[k] = attr_labels[k].to(device)
    pred_id, attrs_pred = apr(img)
    id_loss = loss_fn(pred_id, pid)
    attr_loss = {}
    for attr, pred in attrs_pred.items():
        attr_loss[attr] = loss_fn(pred, attr_labels[attr])

    total_loss = 8 * id_loss + sum(attr_loss.values()) / len(attr_loss)
    loss_t += total_loss.item()

print("###############################")
print(loss_t/len(pd))
