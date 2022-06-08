# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        
        
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        return x

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class DualSegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(DualSegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = BasicConv2d(embedding_dim*4, embedding_dim, 1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        self.up4 = BasicConv2d(
            1024+512, 512,
            kernel_size=3,
            padding=1,
        )
        self.up3 = BasicConv2d(
            640+320, 320,
            kernel_size=3,
            padding=1,
        )
        self.up2 = BasicConv2d(
            320+128, 128,
            kernel_size=3,
            padding=1,
        )
        self.up1 = BasicConv2d(
            128+64, 64,
            kernel_size=3,
            padding=1,
        )

    def forward(self, inputs):
        hardnetout = inputs[0]
        segout = inputs[1]
        x = self._transform_inputs(segout)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x # 
        x1 = hardnetout[0] #
        x2 = hardnetout[1] # 
        x3 = hardnetout[2] # 
        x4 = hardnetout[3] # 

        c1 = torch.cat((x1, c1), 1)
        c1 = self.up1(c1)
        c2 = torch.cat((x2, c2), 1)
        c2 = self.up2(c2)
        c3 = torch.cat((x3, c3), 1)
        c3 = self.up3(c3)
        c4 = torch.cat((x4, c4), 1)
        c4 = self.up4(c4)
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

@HEADS.register_module()
class DualSegFormerHead_ver2(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(DualSegFormerHead_ver2, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        
        self.up4 = BasicConv2d(
            1024+512, 256,
            kernel_size=3,
            padding=1,
        )
        self.up3 = BasicConv2d(
            640+320, 256,
            kernel_size=3,
            padding=1,
        )
        self.up2 = BasicConv2d(
            320+128, 256,
            kernel_size=3,
            padding=1,
        )
        
        self.rfb4 = RFB_modified(256, 32)
        self.rfb3 = RFB_modified(256, 32)
        self.rfb2 = RFB_modified(256, 32)
        self.agg1 = aggregation(32)    

    def forward(self, inputs):
        hardnetout = inputs[0]
        segout = inputs[1]
        x = self._transform_inputs(segout)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x 
        x2 = hardnetout[1] 
        x3 = hardnetout[2] 
        x4 = hardnetout[3] 

        c2 = torch.cat((x2, c2), 1)
        c2 = self.up2(c2)
        c2 = self.rfb2(c2)
        
        c3 = torch.cat((x3, c3), 1)
        c3 = self.up3(c3)
        c3 = self.rfb3(c3)
        
        c4 = torch.cat((x4, c4), 1)
        c4 = self.up4(c4)
        c4 = self.rfb4(c4)
        
        c5 = self.agg1(c4, c3, c2)
        c5 = F.interpolate(c5, scale_factor=4, mode='bilinear')
        
        return c5
        
        
@HEADS.register_module()
class SegFormerHead_ver2(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_ver2, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        
        # self.up4 = BasicConv2d(
        #     512, 256,
        #     kernel_size=3,
        #     padding=1,
        # )
        # self.up3 = BasicConv2d(
        #     320, 256,
        #     kernel_size=3,
        #     padding=1,
        # )
        # self.up2 = BasicConv2d(
        #     128, 256,
        #     kernel_size=3,
        #     padding=1,
        # )
        
        self.rfb2 = RFB_modified(128, 32)
        self.rfb3 = RFB_modified(320, 32)
        self.rfb4 = RFB_modified(512, 32)
        self.agg1 = aggregation(32)    

    def forward(self, inputs):
        segout = inputs
        x = self._transform_inputs(segout)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x 

        # c2 = self.up2(c2)
        c2 = self.rfb2(c2)
        
        # c3 = self.up3(c3)
        c3 = self.rfb3(c3)
        
        # c4 = self.up4(c4)
        c4 = self.rfb4(c4)
        
        c5 = self.agg1(c4, c3, c2)
        c5 = F.interpolate(c5, scale_factor=4, mode='bilinear')
        
        return c5


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x