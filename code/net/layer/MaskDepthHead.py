import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt


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
        self.prob = nn.Softmax()

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
