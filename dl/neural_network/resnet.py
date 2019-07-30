from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch


class ResNet50(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.channel_mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().reshape(1, -1, 1, 1)
        self.channel_std = torch.Tensor([0.229, 0.224, 0.225]).cuda().reshape(1, -1, 1, 1)
        self.ResNet = models.resnet50(pretrained=pretrained)
        # pytorch's standard implementation throws errors at some image sizes..
        self.ResNet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.ResNet.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)


    def forward(self, x):
        """
        Forward pass. Gray image is copied into pseudo 3dim rgb image and mean/std are adapted to
        the ImageNet distribution

        :param x:
        :return:
        """
        # copy to 3 channels
        x = x.repeat(1, 3, 1, 1)
        # substract imagenet mean and scale imagenet std
        x -= self.channel_mean
        x /= self.channel_std
        x = self.ResNet(x)
        return F.log_softmax(x, dim=1)

