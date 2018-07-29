import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from layers.functions.prior_box import PriorBox
from layers.functions.detection import Detect
from layers.modules.l2norm import L2Norm
from data import v2


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # TODO: implement __call__ in PriorBox
        self.priorbox = PriorBox(v2)

        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = 512

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        # fused conv4_3 and conv5_3
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.deconv = nn.ConvTranspose2d(512, 512, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.L2Norm5_3 = L2Norm(512, 10)

        self.conv_cat = nn.Conv2d(1024, 512, 1, 1)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 300, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # Apply vgg up to conv4_3 relu
        # Fused conv4_3,conv5_3
        for k in range(23):
            x = self.vgg[k](x)
        conv4_3 = self.conv4_3(x)
        s4_3 = self.L2Norm(conv4_3)

        for k in range(23, 30):
            x = self.vgg[k](x)
        deconv = self.deconv(x)
        conv5_3 = self.conv5_3(deconv)
        s5_3 = self.L2Norm5_3(conv5_3)

        # s = s4_3 + s5_3
        # s = self.fused_relu(s)
        s = torch.cat([F.relu(s4_3), F.relu(s5_3)], dim=1)
        s = F.relu(self.conv_cat(s))
        sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # Apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # Apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            conf_preds = conf.view(-1, self.num_classes)
            conf_preds = self.softmax(conf_preds).view(conf.size(0), -1, self.num_classes)
            # TODO 测试
            # loc = loc.view(loc.size(0), -1, 4)
            # print(loc.size())
            # print(conf_preds.size())
            # print(self.priors.size())
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
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')

        else:
            print("Sorry only .pth or .pkl files supported.")

    def load_weights_fused(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            params = torch.load(base_file, map_location=lambda storage, loc: storage)
            own_dict = self.state_dict()
            for k, v in list(own_dict.items())[:51]:
                param = params.get(k)
                if param is None:
                    continue
                v.copy_(param)
            print('Finished!')

        else:
            print("Sorry only .pth or .pkl files supported.")

    def load_weights_for_rosd(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            params = torch.load(base_file, map_location=lambda storage, loc: storage)
            own_dict = self.state_dict()
            for k, v in list(own_dict.items())[:-28]:
                param = params.get(k)
                if param is None:
                    continue
                v.copy_(param)
            print('Finished!')

        else:
            print("Sorry only .pth or .pkl files supported.")


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False

    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 4)[flag], stride=1, padding=(1, 0)[flag])]
            else:
                layers += [nn.Conv2d(in_channels, v,
                                     kernel_size=(1, 3)[flag],
                                     stride=(1, 2)[flag],
                                     padding=(1, 0)[flag])]
            flag = not flag
        in_channels = v

    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_sources = [21, -2]

    for k, v in enumerate(vgg_sources):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers)


base = {'512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]}

extras = {'512': [256, 512, 128, 256, 128, 256, 128, 256, 128, 'S', 256]}
mbox = {'512': [4, 6, 6, 6, 6, 4, 4]}  # number of boxes per feature map location


def build_ssd(phase, size=512, num_classes=21):
    if phase != 'test' and phase != 'train':
        print("Error: Phase not recognized.")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)

    return SSD(phase, base_, extras_, head_, num_classes)


if __name__ == '__main__':

    base_, extras_, head_ = multibox(vgg(base[str(512)], 3),
                                     add_extras(extras[str(512)], 1024),
                                     mbox[str(512)], 21)
    for layer in base_:
        print(layer)
    print(len(base_))

    for layer in extras_:
        print(layer)
    print(len(extras_))

    for layer in head_:
        print('')
        for l in layer:
            print(l)
    print(len(head_))
