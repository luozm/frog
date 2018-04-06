import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from net.layer.rpn_multi_nms import make_rpn_windows, rpn_nms
from net.layer.rpn_multi_target import make_rpn_target
from net.layer.rpn_multi_loss import rpn_loss


# -------------------------------------------------------------------------------------------
# Blocks for SE-ResNext-50
#
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py
# -------------------------------------------------------------------------------------------

class ConvBn2d(nn.Module):

    def merge_bn(self):
        assert self.conv.bias is None
        conv_weight = self.conv.weight.data
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var = self.bn.running_var
        bn_eps = self.bn.eps

        #https://github.com/sanghoon/pva-faster-rcnn/issues/5
        #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N, C, KH, KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var+bn_eps))
        std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
        conv_weight_hat = std_bn_weight*conv_weight
        conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)

        self.bn   = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data = conv_bias_hat

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

        if is_bn is False:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class SELayer(nn.Module):
    """SE layer

    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class SENextBottleneckBlock(nn.Module):
    """bottleneck type C

    """

    def __init__(self, in_planes, planes, out_planes, groups, reduction=16, is_downsample=False, stride=1):
        super(SENextBottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        self.conv_bn1 = ConvBn2d(in_planes, planes, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(planes, planes, kernel_size=3, padding=1, stride=stride, groups=groups)
        self.conv_bn3 = ConvBn2d(planes, out_planes, kernel_size=1, padding=0, stride=1)
        self.scale = SELayer(out_planes, reduction)

        if is_downsample:
            self.downsample = ConvBn2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):

        z = F.relu(self.conv_bn1(x), inplace=True)
        z = F.relu(self.conv_bn2(z), inplace=True)
        z = self.conv_bn3(z)

        if self.is_downsample:
            z = self.scale(z) + self.downsample(x)
        else:
            z = self.scale(z) + x

        z = F.relu(z, inplace=True)
        return z


class LateralBlock(nn.Module):
    """P layer

    """
    def __init__(self, c_planes, p_planes, out_planes ):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes,  p_planes,   kernel_size=1, padding=0, stride=1)
        self.top = nn.Conv2d(p_planes,  out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c, p):
        _, _, H, W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2,mode='nearest')
        p = p[:, :, :H, :W] + c
        p = self.top(p)

        return p


def make_layer_c0(in_planes, out_planes):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


def make_layer_c(in_planes, planes, out_planes, groups, num_blocks, stride):

    layers = [SENextBottleneckBlock(in_planes, planes, out_planes, groups, is_downsample=True, stride=stride)]
    for i in range(1, num_blocks):
        layers.append(SENextBottleneckBlock(out_planes, planes, out_planes, groups))

    return nn.Sequential(*layers)


class FeatureNet(nn.Module):
    """ResNext-50 32x4d

    """

    def __init__(self, cfg, in_channels, out_channels=256):
        super(FeatureNet, self).__init__()
        self.cfg = cfg

        # bottom-top
        self.layer_c0 = make_layer_c0(in_channels, 64)
        self.layer_c1 = make_layer_c(64, 64, 256, groups=32, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer_c2 = make_layer_c(256, 128, 512, groups=32, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer_c3 = make_layer_c(512, 256, 1024, groups=32, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer_c4 = make_layer_c(1024, 512, 2048, groups=32, num_blocks=3, stride=2)  #out = 512*4 = 2048

        # top-down
        self.layer_p4 = nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer_p3 = LateralBlock(1024, out_channels, out_channels)
        self.layer_p2 = LateralBlock(512, out_channels, out_channels)
        self.layer_p1 = LateralBlock(256, out_channels, out_channels)

    def forward(self, x):

        c0 = self.layer_c0(x)

        c1 = self.layer_c1(c0)
        c2 = self.layer_c2(c1)
        c3 = self.layer_c3(c2)
        c4 = self.layer_c4(c3)

        p4 = self.layer_p4(c4)
        p3 = self.layer_p3(c3, p4)
        p2 = self.layer_p2(c2, p3)
        p1 = self.layer_p1(c1, p2)

        features = [p1, p2, p3, p4]
        assert(len(self.cfg.rpn_scales) == len(features))

        return features


# -------------------------------------------------------------------------------------------
# Rpn head
# -------------------------------------------------------------------------------------------

class RpnMultiHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RpnMultiHead, self).__init__()

        self.num_scales = len(cfg.rpn_scales)
        self.num_bases = [len(b) for b in cfg.rpn_base_apsect_ratios]

        self.convs = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.deltas = nn.ModuleList()
        for l in range(self.num_scales):
            channels = in_channels*2
            self.convs.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            self.logits.append(
                nn.Sequential(
                    nn.Conv2d(channels, self.num_bases[l]*2,   kernel_size=3, padding=1),
                )
            )
            self.deltas.append(
                nn.Sequential(
                    nn.Conv2d(channels, self.num_bases[l]*2*4, kernel_size=3, padding=1),
                )
            )

    def forward(self, fs):
        batch_size = len(fs[0])

        logits_flat = []
        deltas_flat = []
        for l in range(self.num_scales):  # apply multibox head to feature maps
            f = fs[l]
            f = F.relu(self.convs[l](f))

            f = F.dropout(f, p=0.5, training=self.training)
            logit = self.logits[l](f)
            delta = self.deltas[l](f)

            logit_flat = logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2, 4)
            logits_flat.append(logit_flat)
            deltas_flat.append(delta_flat)

        logits_flat = torch.cat(logits_flat, 1)
        deltas_flat = torch.cat(deltas_flat, 1)

        return logits_flat, deltas_flat


class SingleRPN(nn.Module):

    def __init__(self, cfg):
        super(SingleRPN, self).__init__()
        self.version = 'net version \'rpn-se-resnext50-fpn\''
        self.cfg = cfg
        self.mode = 'train'

        feature_channels = 256
        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.rpn_head = RpnMultiHead(cfg, feature_channels)

    def forward(self, inputs, truth_boxes=None,  truth_labels=None, truth_instances=None):
        cfg = self.cfg
        mode = self.mode
        batch_size = len(inputs)

        # features
        features = data_parallel(self.feature_net, inputs)

        # rpn proposals -------------------------------------------
        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn_head, features)
        self.rpn_window = make_rpn_windows(cfg, features)
        self.rpn_proposals = rpn_nms(cfg, mode, inputs, self.rpn_window, self.rpn_logits_flat, self.rpn_deltas_flat)

        if mode in ['train', 'valid']:
            self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                make_rpn_target(cfg, inputs, self.rpn_window, truth_boxes, truth_labels)

    def loss(self):

        self.rpn_cls_loss, self.rpn_reg_loss = \
           rpn_loss(self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights)

        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss

        return self.total_loss

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
