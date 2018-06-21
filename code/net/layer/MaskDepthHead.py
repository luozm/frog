import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt


class MaskDirDepthHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskDepthHead, self).__init__()
#        self.num_classes = cfg.num_classes

        self.dir_conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        self.dir_bn1 = nn.BatchNorm2d(256)
        self.dir_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.dir_bn2 = nn.BatchNorm2d(256)
        self.dir_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.dir_bn3 = nn.BatchNorm2d(256)
        self.dir_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.dir_bn4 = nn.BatchNorm2d(256)

        self.dir_up = nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.dir_logit = nn.Conv2d(256, 2, kernel_size=1, padding=0, stride=1)

        self.dep_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.dep_bn1 = nn.BatchNorm2d(256)
        self.dep_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.dep_bn2 = nn.BatchNorm2d(256)
        self.dep_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.dep_bn3 = nn.BatchNorm2d(256)
        self.dep_conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.dep_bn4 = nn.BatchNorm2d(256)

        self.dep_up = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False)
        self.dep_logit = nn.Conv2d(256, 16, kernel_size=1, padding=0, stride=1)
        self.dep_prob = nn.Softmax()

    def forward(self, crops):
        x = F.relu(self.dir_bn1(self.dir_conv1(crops)), inplace=True)
        x = F.relu(self.dir_bn2(self.dir_conv2(x)), inplace=True)
        x = F.relu(self.dir_bn3(self.dir_conv3(x)), inplace=True)
        x = F.relu(self.dir_bn4(self.dir_conv4(x)), inplace=True)
        x = self.dir_up(x)
        dir_logits = self.dir_logit(x)

        x = F.relu(self.dep_bn1(self.dep_conv1(x)), inplace=True)
        x = F.relu(self.dep_bn2(self.dep_conv2(x)), inplace=True)
        x = F.relu(self.dep_bn3(self.dep_conv3(x)), inplace=True)
        x = F.relu(self.dep_bn4(self.dep_conv4(x)), inplace=True)
        x = self.dep_up(x)
        dep_logits = self.dep_logit(x)

        _, argmax = torch.max(dep_logits, dim=1)

        #logits = self.prob(logits)

        return dir_logits, dep_logits, argmax


class MaskDepthHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskDepthHead, self).__init__()
#        self.num_classes = cfg.num_classes

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.up = nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.logit = nn.Conv2d(256, 16, kernel_size=1, padding=0, stride=1)

    def forward(self, crops):
        x = F.relu(self.bn1(self.conv1(crops)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.up(x)
        logits = self.logit(x)

        _, argmax = torch.max(logits, dim=1)

        #logits = self.prob(logits)

        return logits, argmax
