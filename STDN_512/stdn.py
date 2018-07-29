import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from layers.functions.prior_box import PriorBox
from layers.functions.detection import Detect
from data import v2
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class STDN(nn.Module):
    def __init__(self, phase, base, head, num_classes):
        super(STDN, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = 512  # 没有用

        self.priorbox = PriorBox(v2)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

        self.basenet = nn.ModuleList(base)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.out1 = nn.AvgPool2d(9, 9)
        self.out2 = nn.AvgPool2d(3, 3)
        self.out3 = nn.AvgPool2d(2, 2)

        if self.phase == 'test':
            self.softmax = nn.Softmax()
            self.detect = Detect(num_classes, 0, 300, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        for i in range(7):
            if i % 2 == 1:
                tmp_layers = self.basenet[i]
                for k in range(len(tmp_layers)):
                    x = tmp_layers[k](x)
            else:
                x = self.basenet[i](x)
        for i in range(7, len(self.basenet)):
            for k in range(5):
                x = self.basenet[i][k](x)
            s = F.relu(self.out1(x))
            sources.append(s)

            for k in range(5, 10):
                x = self.basenet[i][k](x)
            s = F.relu(self.out2(x))
            sources.append(s)

            for k in range(10, 15):
                x = self.basenet[i][k](x)
            s = F.relu(self.out3(x))
            sources.append(s)

            for k in range(15, 20):
                x = self.basenet[i][k](x)
            sources.append(F.relu(x))

            for k in range(20, 25):
                x = self.basenet[i][k](x)
            # TODO 只是reshape了 再看论文中怎么说
            s = x.view(x.size()[0], -1, x.size()[2] * 2, x.size()[3] * 2)
            sources.append(F.relu(s))

            for k in range(25, 32):
                x = self.basenet[i][k](x)
            # TODO 只是reshape了 再看论文中怎么说
            s = x.view(x.size()[0], -1, x.size()[2] * 4, x.size()[3] * 4)
            sources.append(F.relu(s))

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            conf_preds = conf.view(-1, self.num_classes)
            conf_preds = self.softmax(conf_preds).view(conf.size(0), -1, self.num_classes)
            # TODO 进行测试
            # print(conf_preds.size())
            # print(loc.view(loc.size(0), -1, 4).size())
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
            # print(self.state_dict().keys())
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print("Sorry only .pth or .pkl files supported.")

    def load_weights_pretrain(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            params = torch.load(base_file)
            own_dict_value = list(self.state_dict().values())[15:-168]
            params_keys = list(params.keys())[5:-6]
            for k, v in zip(params_keys, own_dict_value):
                v.copy_(params.get(k))
            print('Finished!')
        else:
            print("Sorry only .pth or .pkl files supported.")

    def load_weights_test(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            params = torch.load(base_file)
            own_dict = self.state_dict()
            for k, v in list(own_dict.items())[15:]:
                v.copy_(params.get(k))
            print('Finished!')
        else:
            print("Sorry only .pth or .pkl files supported.")

    def load_weights_voc(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict ...')
            params = torch.load(base_file)
            keys = ['conf.{}.6.weight'.format(i) for i in range(6)]
            keys += ['conf.{}.6.bias'.format(i) for i in range(6)]
            for k, v in self.state_dict().items():
                if k in keys:
                    continue
                v.copy_(params.get(k))
            print('Finished!')
        else:
            print("Sorry only .pth or .pkl files supported.")


def denseblock(num_layers, num_input_features, bn_size, growth_rate, drop_rate):
    layers = []
    for i in range(num_layers):
        layers += [_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)]
    return layers


def densenet(growth_rate=32, block_config=(6, 12, 32, 32),
             num_init_features=64, bn_size=4, drop_rate=0):
    layers = []
    layers += [nn.Sequential(OrderedDict([
        ('norm0', nn.BatchNorm2d(3)),
        ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=2, bias=False)),
        ('norm1', nn.BatchNorm2d(num_init_features)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv1', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=2, bias=False)),
        ('norm2', nn.BatchNorm2d(num_init_features)),
        ('relu2', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=2, bias=False)),
        ('pool0', nn.AvgPool2d(kernel_size=2, stride=2)),
    ]))]
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        block = denseblock(num_layers=num_layers, num_input_features=num_features,
                           bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        layers += [nn.ModuleList(block)]
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            layers += [trans]
            num_features = num_features // 2

    return layers


# TODO 最后的分类器和坐标偏移的网络中channels没说明白
def multibox(basenet, num_classes):
    loc_layers = []
    conf_layers = []
    in_channels = [800, 960, 1120, 1280, 360, 104]

    for in_channel in in_channels:
        loc_layers += [nn.Sequential(nn.Conv2d(in_channel, in_channel // 2, kernel_size=1),
                                     nn.BatchNorm2d(in_channel // 2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(in_channel // 2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channel // 2, 8 * 4, kernel_size=3, padding=1))]

        conf_layers += [nn.Sequential(nn.Conv2d(in_channel, in_channel // 2, kernel_size=1),
                                      nn.BatchNorm2d(in_channel // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channel // 2, in_channel // 2, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(in_channel // 2),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channel // 2, 8 * num_classes, kernel_size=3, padding=1))]

        # loc_layers += [nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1),
        #                              nn.BatchNorm2d(in_channel),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
        #                              nn.BatchNorm2d(in_channel),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(in_channel, 8 * 4, kernel_size=3, padding=1))]
        #
        # conf_layers += [nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=1),
        #                               nn.BatchNorm2d(in_channel),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
        #                               nn.BatchNorm2d(in_channel),
        #                               nn.ReLU(inplace=True),
        #                               nn.Conv2d(in_channel, 8 * num_classes, kernel_size=3, padding=1))]
    return loc_layers, conf_layers


def build_stdn(phase='train', num_classes=21):
    if phase != 'test' and phase != 'train':
        print("Error: Phase not recognized.")
        return
    basenet = densenet()
    head = multibox(basenet, num_classes)

    stdn = STDN(phase, basenet, head, num_classes)
    return stdn


if __name__ == '__main__':
    model = build_stdn()
    # print(model)
    x = torch.randn(1, 3, 513, 513)
    from torch.autograd import Variable

    x = Variable(x)
    model(x)
    print(model.loc)
    print(model.conf)
