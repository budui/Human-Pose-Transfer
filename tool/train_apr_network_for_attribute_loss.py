import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import init
from torchvision import models
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, Timer

from scipy.io import loadmat


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


class AttrMarket1501(Dataset):
    def __init__(self, root_path, attr_path, transfrom=None, loader=default_loader):
        self.path = root_path
        self.attr_path = attr_path
        if not os.path.isdir(self.path):
            raise NotADirectoryError(self.path)

        self.transform = transfrom
        self.loader = loader

        self.mat = loadmat(self.attr_path)["market_attribute"][0][0]
        self.attrs = self._make_attr_dict("train")

        file_list = os.listdir(os.path.join(self.path, "bounding_box_train"))

        self.file_list = []
        for f in file_list:
            if f.endswith(".jpg"):
                self.file_list.append(f)
        self.name_to_class = self._id_name_to_class()

    def _id_name_to_class(self):
        names = set(sorted([fn[:4] for fn in self.file_list]))
        return dict(zip(names, range(len(names))))

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

    def __getitem__(self, index):
        img_path = self.file_list[index]
        id_name = img_path[:4]
        attr = self.attrs[id_name]
        class_id = self.name_to_class[id_name]
        image = self.loader(os.path.join(self.path, "bounding_box_train/", img_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, class_id, attr

    def __repr__(self):
        r = "<AttrMarket1501 len: {} ids:{}>".format(len(self), len(self.name_to_class))
        return r


def get_trainer(opt, device="cuda"):
    apr = APR(dict(zip(ATTR_NAMES, ATTR_NUM_CLASS)), 751)
    print(apr)
    apr.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    apr_classifier_params = list(apr.id_classifier.parameters())
    for c in apr.attr_classifier.values():
        apr_classifier_params += list(c.parameters())
    ignored_params = list(map(id, apr_classifier_params))
    base_params = filter(lambda p: id(p) not in ignored_params, apr.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': apr_classifier_params, 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    def step(engine, batch):
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

        total_loss = 8 * id_loss + sum(attr_loss.values())/len(attr_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        output = dict(loss={}, pred={}, real={})
        output["loss"]["id"] = id_loss.item()
        output["loss"].update({a: l.item() for a, l in attr_loss.items()})
        output["pred"]["id"] = pred_id
        output["pred"].update({a: p.data for a, p in attrs_pred.items()})
        output["real"]["id"] = pid
        output["real"].update({a: l.data for a, l in attr_labels.items()})
        return output

    trainer = Engine(step)

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        exp_lr_scheduler.step()

    Accuracy(output_transform=lambda o: [o['pred']["id"], o["real"]["id"]]).attach(trainer, "acc_id")
    Accuracy(output_transform=lambda o: [o['pred']["up"], o["real"]["up"]]).attach(trainer, "acc_up")
    Accuracy(output_transform=lambda o: [o['pred']["gender"], o["real"]["gender"]]).attach(trainer, "acc_gender")
    Accuracy(output_transform=lambda o: [o['pred']["downpink"], o["real"]["downpink"]]).attach(trainer, "acc_downpink")
    Accuracy(output_transform=lambda o: [o['pred']["bag"], o["real"]["bag"]]).attach(trainer, "acc_bag")
    Accuracy(output_transform=lambda o: [o['pred']["age"], o["real"]["age"]]).attach(trainer, "acc_age")
    Accuracy(output_transform=lambda o: [o['pred']["backpack"], o["real"]["backpack"]]).attach(trainer, "acc_backpack")
    Accuracy(output_transform=lambda o: [o['pred']["uppurple"], o["real"]["uppurple"]]).attach(trainer, "acc_uppurple")
    Accuracy(output_transform=lambda o: [o['pred']["hair"], o["real"]["hair"]]).attach(trainer, "acc_hair")
    Accuracy(output_transform=lambda o: [o['pred']["clothes"], o["real"]["clothes"]]).attach(trainer, "acc_clothes")
    Accuracy(output_transform=lambda o: [o['pred']["down"], o["real"]["down"]]).attach(trainer, "acc_down")
    Accuracy(output_transform=lambda o: [o['pred']["handbag"], o["real"]["handbag"]]).attach(trainer, "acc_handbag")

    RunningAverage(output_transform=lambda o: o["loss"]["id"]).attach(trainer, "loss_id")
    for an in ATTR_NAMES:
        RunningAverage(output_transform=lambda o: o["loss"][an]).attach(trainer, "loss_{}".format(an))

    monitoring_metrics = ['loss_id', 'loss_gender', 'loss_backpack']

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    checkpoint_handler = ModelCheckpoint(
        opt.output_dir, 'networks',
        save_interval=opt.save_interval, n_saved=opt.n_saved,
        require_empty=False, create_dir=True, save_as_state_dict=True
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, to_save={"APR": apr})

    def print_log(engine):
        pbar.log_message('Epoch {} done. Time: {:.3f}[batch/s]*{}[batch] = {:.3f}[s]'.format(
            engine.state.epoch,
            timer.value(),
            engine.state.iteration,
            timer.value() * engine.state.iteration
        ))
        timer.reset()
        opf = ""
        for m, v in engine.state.metrics.items():
            if "loss" in m:
                continue
            opf += "{}:{:2.4f} ".format(m,v)
        pbar.log_message(opf)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_log)

    return trainer


def get_data_loader(opt, device="cuda"):
    dataset = AttrMarket1501(
        opt.market1501,
        "data/market/attribute/market_attribute.mat",
        transforms.Compose([
            transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    print(dataset)
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                       shuffle=True, num_workers=8, pin_memory=True, drop_last=True)


def main():
    parser = ArgumentParser(description="Training")
    parser.add_argument('--gpu_id', default=1, type=int, help='gpu_id: e.g. 0')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument("--epochs", default=80, type=int, help="epoch_num")
    parser.add_argument("--output_dir", type=str, default="attr_loss/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--save_interval", default=5, type=int, help="models will be saved to disk "
                                                                        "every save_interval calls to the handler.")
    parser.add_argument("--n_saved", default=10, type=int, help="Number of models that should be kept on disk. "
                                                                "Older files will be removed.")
    parser.add_argument('--market1501', type=str, default="/data/Market-1501-v15.09.15/")

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)
    cudnn.benchmark = True

    trainer = get_trainer(opt)
    data_loader = get_data_loader(opt)
    trainer.run(data_loader, max_epochs=opt.epochs)


if __name__ == '__main__':
    main()