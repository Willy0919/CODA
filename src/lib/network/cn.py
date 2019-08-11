#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Willy
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class CNet(nn.Module):
    def __init__(self, in_place):
        super(CNet,self).__init__()
        self.in_place = in_place
        self.fc6 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.fc7 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=4, dilation=4)

        self.densitymap = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, dilation=1)

        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear')



    def forward(self,x):
        for i in range(len(self.CNN_base)):
            x = self.CNN_base[i](x)
            if i == 22:
                x4_3 = x

            # print(x.size())
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.densitymap(x)

        x = self.upsample(x) / 16

        return x


    def _init_weights(self,truncate=False):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.fc6, 0, 0.01, truncate)
        normal_init(self.fc7, 0, 0.01, truncate)
        normal_init(self.densitymap, 0, 0.01, truncate)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


class vgg16(CNet):
    name = 'CNet_VGG'
    def __init__(self, pretrained=True):
        self.model_path = 'data/pretrained_models/vgg16_caffe.pth'
        self.pretrained = pretrained
        CNet.__init__(self,in_place=256)


    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print('Loading pretrained weights from %s'%(self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

        #modified conv to dilate
        # features = list(vgg.features.children())
        # features_d = [features[i] for i in range(23)]
        # for i in range(24,30):
        #     features_d.append(features[i])
        # self.CNN_base = nn.Sequential(*(features_d))
        # for i in [23,25,27]:
        #     self.CNN_base[i].dilation = (2,2)
        #     self.CNN_base[i].padding =(2,2)

        self.CNN_base = nn.Sequential(*list(vgg.features._modules.values())[:30])
        # for i, k in enumerate(self.CNN_base):
        #     print(i,k)
        #fixed the layers before conv3
        for layer in range(10):
            for p in self.CNN_base[layer].parameters(): p.requires_grad = False

if __name__ == '__main__':
    cn = vgg16()
    cn.create_architecture()
    for k,v in cn.state_dict().items():
        print(k)



