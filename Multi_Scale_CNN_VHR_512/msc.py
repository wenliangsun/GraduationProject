import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from layers.functions.prior_box import PriorBox
from layers.functions.detection import Detect
from layers.modules.l2norm import L2Norm
from data import v2

base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]


class MSC(nn.Module):
    def __init__(self, phase, base, head, num_classes):
        super(MSC, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 512
        self.vgg = nn.ModuleList(base)

        self.L2Norm4_3 = L2Norm(512, 20)
        self.L2Norm5_3 = L2Norm(512, 10)

        self.deconv7 = nn.ConvTranspose2d(2048, 2048, 2, 2)
        self.deconv6 = nn.ConvTranspose2d(2048, 512, 2, 2)
        self.deconv5 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.conv_fc6 = nn.Conv2d(2048, 2048, 3, 1, 1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 300, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        for k in range(23):
            x = self.vgg[k](x)
        conv4_3 = self.conv4_3(x)

        for k in range(23, 30):
            x = self.vgg[k](x)
        conv5_3 = self.conv5_3(x)

        for k in range(30, 33):
            x = self.vgg[k](x)
        conv_fc6 = self.conv_fc6(x)

        for k in range(33, 36):
            x = self.vgg[k](x)
        f7 = F.relu(x)
        deconv7 = self.deconv7(f7)
        f6 = F.relu(conv_fc6 + deconv7)

        deconv6 = self.deconv6(f6)
        f5 = F.relu(conv5_3 + deconv6)
        # f5 = self.L2Norm5_3(f5)

        deconv5 = self.deconv5(f5)
        f4 = F.relu(self.L2Norm4_3(conv4_3) + deconv5)
        # f4 = self.L2Norm4_3(f4)

        sources.extend([f4, f5, f6, f7])

        # Apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            conf_preds = conf.view(-1, self.num_classes)
            conf_preds = self.softmax(conf_preds).view(conf.size(0), -1, self.num_classes)
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                conf_preds,
                self.priors.type(type(x.data))  # default boxes
            )
        else:
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes),
                      self.priors)

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            # print(self.state_dict().keys())
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print("Sorry only .pth or .pkl files supported.")

    def load_weights_pretrain(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            params = torch.load(base_file, map_location=lambda storage, loc: storage)
            own_dict = self.state_dict()
            for k, v in list(own_dict.items())[:-16]:
                param = params.get(k)
                if param is None:
                    continue
                v.copy_(param)
            print('Finished!')

        else:
            print("Sorry only .pth or .pkl files supported.")

    def load_weights_voc(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            params = torch.load(base_file, map_location=lambda storage, loc: storage)
            own_dict = self.state_dict()
            for k, v in list(own_dict.items())[:-16]:
                param = params.get(k)
                if param is None:
                    continue
                v.copy_(param)
            print('Finished!')

        else:
            print("Sorry only .pth or .pkl files supported.")


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
    fc6 = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
    pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
    fc7 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)

    layers += [pool5, fc6, nn.ReLU(inplace=True), pool6, fc7, nn.ReLU(inplace=True)]

    return layers


def multibox(num_classes):
    loc_layers = []
    conf_layers = []
    in_channels = [512, 512, 2048, 2048]
    for in_channel in in_channels:
        loc_layers += [nn.Conv2d(in_channel, 8 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channel, 8 * num_classes, kernel_size=3, padding=1)]

    return loc_layers, conf_layers


def build_msc(phase='train', num_classes=21):
    if phase != 'test' and phase != 'train':
        print("Error: Phase not recognized.")
        return
    head = multibox(num_classes)

    return MSC(phase, vgg(base, 3), head, num_classes)


if __name__ == '__main__':
    net = build_msc('train', 21)
    print(net)
    aa = torch.randn(1, 3, 513, 513)
    aa = Variable(aa, volatile=True)
    print(net(aa))
