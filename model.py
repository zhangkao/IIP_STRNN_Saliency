import torch
import torch.nn as nn
import torch.nn.functional as F
import os, math, numpy

from model_dcn_resnet import *
from model_dcn_vgg16 import *
from model_pwc import *
from model_convlstm import *
from model_att import *

feature_loader = {
    'vgg16': vgg16,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

sfnet_inplanes = {
    'vgg16': [128, 256, 512, 512],
    'resnet18': [64, 128, 256, 512],
    'resnet34': [64, 128, 256, 512],
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048],
}

init_func = {
    'uniform': nn.init.uniform_,
    'normal': nn.init.normal_,
    'constant': nn.init.constant_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_,
    'ones': nn.init.ones_,
    'zeros': nn.init.zeros_,
}


def init_weights(model, funcname='xavier_uniform'):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            init_func[funcname](m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0.)


def normalize_data(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    ims = data.clone()
    ims[:, 0, :, :] = (ims[:, 0, :, :] - mean[0]) / std[0]
    ims[:, 1, :, :] = (ims[:, 1, :, :] - mean[1]) / std[1]
    ims[:, 2, :, :] = (ims[:, 2, :, :] - mean[2]) / std[2]

    return ims


################################################################
# Static Feature Extraction Models
################################################################
class aspp_v3_block(nn.Module):
    def __init__(self, inplanes, planes=256, kernel_size=3, dilation=1):
        super(aspp_v3_block, self).__init__()

        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                                            padding=int(dilation * (kernel_size - 1) / 2), dilation=dilation)
        self.batch_norm = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class backbone_aspp(nn.Module):
    def __init__(self, cnn_type='vgg16',
                 use_dcn=True,
                 strides=[1, 2, 2, 1],
                 dilations=[1, 1, 1, 2],
                 MG_rates=[1, 2, 4],
                 aspp_dilations=[1, 6, 12, 18]):
        super(backbone_aspp, self).__init__()

        if not use_dcn:
            dilations = [1, 1, 1, 1]
            print("use_dcn == False")

        if cnn_type.lower() in ['vgg16']:
            print("cnn_type == 'vgg16'")

        self.feature = feature_loader[cnn_type.lower()](pretrained=True, strides=strides, dilations=dilations,
                                                        MG_rates=MG_rates)
        # for p in self.feature.parameters():
        # 	p.requires_grad = False

        inplanes = sfnet_inplanes[cnn_type][-1]
        planes = 256
        self.aspp1 = aspp_v3_block(inplanes, planes, 1, dilation=aspp_dilations[0])
        self.aspp2 = aspp_v3_block(inplanes, planes, 3, dilation=aspp_dilations[1])
        self.aspp3 = aspp_v3_block(inplanes, planes, 3, dilation=aspp_dilations[2])
        self.aspp4 = aspp_v3_block(inplanes, planes, 3, dilation=aspp_dilations[3])
        self.aspp5 = aspp_v3_block(planes * 4, planes, 1)

        self.conv_low3 = aspp_v3_block(sfnet_inplanes[cnn_type][-3], 128, 1)
        self.conv_low4 = aspp_v3_block(sfnet_inplanes[cnn_type][-2], 128, 1)

        self.conv_last = aspp_v3_block(planes + 128 * 2, planes, 3)

    def forward(self, x):
        _, _, c3, c4, c5 = self.feature(x)

        x1 = self.aspp1(c5)
        x2 = self.aspp2(c5)
        x3 = self.aspp3(c5)
        x4 = self.aspp4(c5)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.aspp5(x)

        x = F.interpolate(out, size=c3.size()[2:], mode='bilinear', align_corners=True)
        low_c3 = self.conv_low3(c3)
        low_c4 = self.conv_low4(c4)
        low_c4 = F.interpolate(low_c4, size=c3.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x, low_c4, low_c3), dim=1)
        x = self.conv_last(x)

        return x


class STRNN_St_Net(nn.Module):
    def __init__(self, cnn_type='vgg16',
                 use_dcn=True,
                 cnn_stride=16,
                 pre_sf_path=''):
        super(STRNN_St_Net, self).__init__()

        if cnn_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            aspp_dilations = [1, 6, 12, 18]
        elif cnn_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
            aspp_dilations = [1, 12, 24, 36]
        else:
            raise ValueError

        sfnet = backbone_aspp(cnn_type=cnn_type, use_dcn=use_dcn, strides=strides, dilations=dilations,
                              MG_rates=[1, 2, 4], aspp_dilations=aspp_dilations)

        if os.path.exists(pre_sf_path):
            print("Load pre-trained SF-Net weights")
            sfnet.load_state_dict(torch.load(pre_sf_path).sfnet.sfnet.state_dict())
        self.sfnet = sfnet

    def forward(self, x):
        x = self.sfnet(x)
        return x


class STRNN_Static_Net(nn.Module):
    def __init__(self, cnn_type='vgg16',
                 use_dcn=True,
                 use_cb=False,
                 use_bn=False,
                 time_dims=7,
                 cnn_stride=16,
                 nb_gaussian=8,
                 pre_sf_path=''):
        super(STRNN_Static_Net, self).__init__()

        self.time_dims = time_dims
        self.use_cb = use_cb
        self.sfnet = STRNN_St_Net(cnn_type, use_dcn, cnn_stride, pre_sf_path)

        if use_bn:
            self.st_layer = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )

            self.dy_layer = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
            )
        else:
            self.st_layer = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.LayerNorm([256, 60, 80]),
                nn.ReLU(),
            )

            self.dy_layer = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.LayerNorm([256, time_dims, 60, 80]),
                nn.ReLU(),
            )

        if use_cb:
            cb_planes = 64
            self.nb_gaussian = nb_gaussian
            self.st_cb_layer = nn.Sequential(
                nn.Conv2d(nb_gaussian, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cb_planes, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cb_planes + 256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.dy_cb_layer = nn.Sequential(
                nn.Conv2d(nb_gaussian, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cb_planes, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(cb_planes + 256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            init_weights(self.st_cb_layer)
            init_weights(self.dy_cb_layer)

        self.conv_out_st = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.conv_out_dy = nn.Conv3d(256, 1, kernel_size=3, padding=1)

        init_weights(self.st_layer)
        init_weights(self.conv_out_st)
        init_weights(self.dy_layer)
        init_weights(self.conv_out_dy)

    def forward(self, x, cb):

        x_sf = self.sfnet(x)
        x_st = self.st_layer(x_sf)

        B_D, C_sf, H, W = x_sf.size()
        B = B_D // self.time_dims
        D = self.time_dims

        x_sf = x_sf.contiguous().view(B, D, C_sf, H, W)
        x_sf = x_sf.permute(0, 2, 1, 3, 4)
        x_dy = self.dy_layer(x_sf)

        if self.use_cb:
            cb_st = cb[:, :self.nb_gaussian, :, :]
            cb_dy = cb[:, self.nb_gaussian:, :, :]

            cb_st = self.st_cb_layer[0:4](cb_st)
            x_st = torch.cat((x_st, cb_st), 1)
            x_st = self.st_cb_layer[4:](x_st)

            cb_dy = self.dy_cb_layer[0:4](cb_dy)
            cb_dy = cb_dy.contiguous().view(B, D, cb_dy.size(1), H, W)
            cb_dy = cb_dy.permute(0, 2, 1, 3, 4)

            x_dy = torch.cat((x_dy, cb_dy), 1)
            x_dy = self.dy_cb_layer[4:](x_dy)

        out_st = F.relu(self.conv_out_st(x_st))
        # out_st = out_st.contiguous().view(B, D, out_st.size(1), H_sf, W_sf)

        out_dy = F.relu(self.conv_out_dy(x_dy))
        out_dy = out_dy.permute(0, 2, 1, 3, 4)
        out_dy = out_dy.contiguous().view(B_D, out_dy.size(2), H, W)

        x_dy = x_dy.permute(0, 2, 1, 3, 4)
        x_dy = x_dy.contiguous().view(B_D, x_dy.size(2), H, W)

        return out_st, x_st, out_dy, x_dy


################################################################
# Dynamic Feature Extraction Models
################################################################
class backbone_pwc(nn.Module):
    def __init__(self, planes=[256, 256, 256],
                 use_flow=False,
                 pre_pwc_path=''):
        super(backbone_pwc, self).__init__()

        self.use_flow = use_flow
        self.feature = PWC_Net()
        if os.path.exists(pre_pwc_path):
            print("Load pre-trained PWC weights")
            self.feature.load_state_dict(torch.load(pre_pwc_path))

        for p in self.feature.parameters():
            p.requires_grad = False

        inplanes = 597
        if use_flow:
            inplanes = 607

        self.layer1 = nn.Sequential(
            nn.Conv2d(inplanes, planes[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(planes[0], planes[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(planes[0], planes[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(planes[1], planes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(planes[1], planes[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(planes[2], planes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(planes[2], planes[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        init_weights(self.layer1)
        init_weights(self.layer2)
        init_weights(self.layer3)

    def forward(self, im1, im2):

        flow, x_pwc = self.feature(im1, im2)

        if self.use_flow:
            flow_abs = torch.abs(flow)
            flow_p = F.relu(flow)
            flow_m = flow_abs - flow_p

            flow_v = torch.sqrt(flow[:, 0:1, :, :] * flow[:, 0:1, :, :] + flow[:, 1:, :, :] * flow[:, 1:, :, :])
            flow_ang = torch.atan2(flow[:, 1:, :, :], flow[:, 0:1, :, :]) / numpy.pi + 1

            flow = torch.cat((flow, flow_p, flow_m, flow_abs, flow_v, flow_ang), 1)
            x_pwc = torch.cat((x_pwc, flow), 1)

        x1 = self.layer1(x_pwc)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x1, x2, x3


class STRNN_OF_Net(nn.Module):
    def __init__(self, use_flow=False,
                 out_stride=8,
                 aspp_dilations=[1, 6, 12, 18],
                 pre_pwc_path=''):
        super(STRNN_OF_Net, self).__init__()

        inplanes = [256, 256, 256]
        self.out_stride = out_stride
        self.ofnet = backbone_pwc(planes=inplanes, use_flow=use_flow, pre_pwc_path=pre_pwc_path)

        planes = 256
        self.aspp1 = aspp_v3_block(inplanes[2], planes, 1, dilation=aspp_dilations[0])
        self.aspp2 = aspp_v3_block(inplanes[2], planes, 3, dilation=aspp_dilations[1])
        self.aspp3 = aspp_v3_block(inplanes[2], planes, 3, dilation=aspp_dilations[2])
        self.aspp4 = aspp_v3_block(inplanes[2], planes, 3, dilation=aspp_dilations[3])
        self.aspp5 = aspp_v3_block(planes * 4, planes, 1)

        self.conv_low1 = aspp_v3_block(inplanes[0], 128, 1)
        self.conv_low2 = aspp_v3_block(inplanes[1], 128, 1)

        self.conv_last = aspp_v3_block(planes + 128 * 2, planes, 3)

        init_weights(self.aspp1)
        init_weights(self.aspp2)
        init_weights(self.aspp3)
        init_weights(self.aspp4)
        init_weights(self.aspp5)
        init_weights(self.conv_low1)
        init_weights(self.conv_low2)
        init_weights(self.conv_last)

    def forward(self, x):
        im1 = x[:-1, :, :, :]
        im2 = x[1:, :, :, :]

        o_h, o_w = im1.shape[2:]
        r_h = int(o_h / self.out_stride)
        r_w = int(o_w / self.out_stride)
        i_h = int(math.floor(math.ceil(o_h / 64.0) * 64.0))
        i_w = int(math.floor(math.ceil(o_w / 64.0) * 64.0))

        in_im1 = F.interpolate(im1, size=(i_h, i_w), mode='bilinear', align_corners=True)
        in_im2 = F.interpolate(im2, size=(i_h, i_w), mode='bilinear', align_corners=True)

        c1, c2, c3 = self.ofnet(in_im1, in_im2)

        x1 = self.aspp1(c3)
        x2 = self.aspp2(c3)
        x3 = self.aspp3(c3)
        x4 = self.aspp4(c3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        out = self.aspp5(x)

        x = F.interpolate(out, size=(r_h, r_w), mode='bilinear', align_corners=True)
        low_c1 = self.conv_low1(c1)
        low_c2 = self.conv_low2(c2)
        low_c1 = F.interpolate(low_c1, size=(r_h, r_w), mode='bilinear', align_corners=True)
        low_c2 = F.interpolate(low_c2, size=(r_h, r_w), mode='bilinear', align_corners=True)

        x = torch.cat((x, low_c1, low_c2), dim=1)
        x = self.conv_last(x)

        return x


class STRNN_Dynamic_Net(nn.Module):
    def __init__(self, out_stride=8,
                 use_flow=False,
                 use_cb=False,
                 use_bn=False,
                 time_dims=7,
                 nb_gaussian=8,
                 pre_of_path=''):
        super(STRNN_Dynamic_Net, self).__init__()

        self.time_dims = time_dims
        self.use_cb = use_cb
        self.ofnet = STRNN_OF_Net(use_flow=use_flow, out_stride=out_stride, pre_pwc_path=pre_of_path)

        if use_bn:
            self.of_layer = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )

            self.of_layer_3d = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(),
            )
        else:
            self.of_layer = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.LayerNorm([256, 60, 80]),
                nn.ReLU(),
            )

            self.of_layer_3d = nn.Sequential(
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(256, 256, kernel_size=3, padding=1),
                nn.LayerNorm([256, time_dims, 60, 80]),
                nn.ReLU(),
            )

        if use_cb:
            cb_planes = 64
            self.of_cb_layer = nn.Sequential(
                nn.Conv2d(nb_gaussian, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cb_planes, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cb_planes + 256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.of_cb_layer_3d = nn.Sequential(
                nn.Conv2d(nb_gaussian, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cb_planes, cb_planes, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(cb_planes + 256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            init_weights(self.of_cb_layer)
            init_weights(self.of_cb_layer_3d)

        self.conv_out_of = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.conv_out_of_3d = nn.Conv3d(256, 1, kernel_size=3, padding=1)

        init_weights(self.of_layer)
        init_weights(self.conv_out_of)
        init_weights(self.of_layer_3d)
        init_weights(self.conv_out_of_3d)

    def forward(self, x, cb):

        x_feat = self.ofnet(x)
        x_of = self.of_layer(x_feat)

        B_D, C, H, W = x_feat.size()
        B = B_D // self.time_dims
        x_feat = x_feat.contiguous().view(B, self.time_dims, C, H, W)
        x_feat = x_feat.permute(0, 2, 1, 3, 4)
        x_of_3d = self.of_layer_3d(x_feat)

        if self.use_cb:
            cb_of = cb[:, :self.nb_gaussian, :, :]
            cb_of_3d = cb[:, self.nb_gaussian:, :, :]

            cb_of = self.of_cb_layer[0:4](cb_of)
            x_of = torch.cat((x_of, cb_of), 1)
            x_of = self.of_cb_layer[4:](x_of)

            cb_of_3d = self.of_cb_layer_3d[0:4](cb_of_3d)
            _, C, H, W = cb_of_3d.size()
            cb_of_3d = cb_of_3d.contiguous().view(B, self.time_dims, C, H, W)
            cb_of_3d = cb_of_3d.permute(0, 2, 1, 3, 4)

            x_of_3d = torch.cat((x_of_3d, cb_of_3d), 1)
            x_of_3d = self.of_cb_layer_3d[4:](x_of_3d)

        out_of = F.relu(self.conv_out_of(x_of))

        out_of_3d = F.relu(self.conv_out_of_3d(x_of_3d))
        out_of_3d = out_of_3d.permute(0, 2, 1, 3, 4)
        out_of_3d = out_of_3d.contiguous().view(B_D, out_of_3d.size(2), H, W)

        x_of_3d = x_of_3d.permute(0, 2, 1, 3, 4)
        x_of_3d = x_of_3d.contiguous().view(B_D, x_of_3d.size(2), H, W)

        return out_of, x_of, out_of_3d, x_of_3d


################################################################
# Saliency Models
#
################################################################
class STRNN_final(nn.Module):
    def __init__(self, ratio=16,
                 pool_type='att',
                 fusion_type=['channel_sr', 'feat_sum'],
                 cnn_type='vgg16',
                 use_dcn=True,
                 use_cb=False,
                 use_bn=False,
                 nb_gaussian=8,
                 cnn_stride=16,
                 out_stride=8,
                 iosize=[480, 640, 60, 80],
                 time_dims=7,
                 cat_type=[0, 1, 0, 1],
                 pre_model_path=''):
        super(STRNN_final, self).__init__()

        self.time_dims = time_dims
        self.use_cb = use_cb
        self.nb_gaussian = nb_gaussian
        self.cat_type = cat_type

        # 1 Saliency Related Feature Extraction Module
        self.feat_sm = STRNN_Static_Net(cnn_type=cnn_type, use_dcn=use_dcn, use_cb=use_cb, use_bn=use_bn,
                                        time_dims=time_dims, cnn_stride=cnn_stride, nb_gaussian=nb_gaussian,
                                        pre_sf_path='')
        self.feat_of = STRNN_Dynamic_Net(out_stride=out_stride, use_flow=False, use_cb=use_cb,
                                         use_bn=use_bn, time_dims=time_dims, nb_gaussian=nb_gaussian,
                                         pre_of_path='')

        # 2 SR-Fu module, ratio = 16 same as SE-Block [70]
        # SE (CRM): pool_type = 'avg'; fusion_type = ['channel_mul']
        # GCM+CRM: pool_type = 'att' and fusion_type = ['channel_mul']
        # CRM+SRM: pool_type = 'avg' and fusion_type = ['channel_sr', 'feat_sum']
        # SR-Fu (GCM+CRM+SRM): pool_type = 'att'; fusion_type = ['channel_sr', 'feat_sum']
        expansion = numpy.sum(cat_type)
        self.att_channel = att_SR(256 * expansion, 256 * expansion // ratio, num_feat=expansion, pool=pool_type,
                                  fusions=fusion_type)

        self.feat_fu = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 3 AA-LSTM module, for ablation snalysis: change 'AttConvLSTM' --> 'ConvLSTM'
        _, _, shape_r_out, shape_c_out = iosize
        self.att_lstm = AttConvLSTM((shape_r_out, shape_c_out), 256, 256, 256, kernel_size=(3, 3), num_layers=1,
                                    batch_first=True, bias=True, return_all_layers=False)

        self.conv_out = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
        )

        if os.path.exists(pre_model_path):
            print("Load pre-trained STRNN weights")
            self.load_state_dict(torch.load(pre_model_path).state_dict(), strict=False)

    def forward(self, x, cb, in_state=None):
        x_sm = normalize_data(x[:-1])
        x_of = x[:, [2, 1, 0], :, :]

        out_sf, x_sf, out_mf, x_mf = self.feat_sm(x_sm, cb)
        out_of, x_of, out_of_3d, x_of_3d = self.feat_of(x_of, cb)

        x_feat = [x_sf, x_mf, x_of, x_of_3d]
        x_cat = [x_feat[i] for i in range(4) if self.cat_type[i] == 1]
        x_fu = torch.cat(x_cat, 1)

        x_fu = self.att_channel(x_fu)
        x_fu = self.feat_fu(x_fu)

        B_D, C, H, W = x_fu.size()
        x_fu = x_fu.contiguous().view(1, B_D, C, H, W)
        x_fu, x_state = self.att_lstm(x_fu, in_state)
        x_fu = x_fu.contiguous().view(B_D, x_fu.size(2), H, W)

        out = F.relu(self.conv_out(x_fu))
        return out, x_state


################################################################
# Ablation Study Models
# For TABLE I
# 1 St-Net: STRNN_Static_Net --> out_st
# 2 OF-Net: STRNN_Dynamic_Net --> out_of
# 3 SR-Fu: STRNN_SRFu_woAALSTM --> out
# 4 STRNN (SR-Fu + AA-LSTM): STRNN_final --> out
#
# For TABLE II, 'STRNN_simFu' and 'STRNN_SRFu_woAALSTM'
# Add-Fu: fu_type='sum'
# SE (CRM): pool_type = 'avg'; fusion_type = ['channel_mul']
# GCM+CRM: pool_type = 'att' and fusion_type = ['channel_mul']
# CRM+SRM: pool_type = 'avg' and fusion_type = ['channel_sr', 'feat_sum']
# SR-Fu (GCM+CRM+SRM): pool_type = 'att'; fusion_type = ['channel_sr', 'feat_sum']
#
# For TABLE III,
# LSTM: STRNN_SRFu_LSTM
# AA-LSTM: STRNN_final
# Bi-AA-LSTM: STRNN_final_biAALSTM
################################################################

class STRNN_SRFu_woAALSTM(nn.Module):
    def __init__(self, ratio=16,
                 pool_type='att',
                 fusion_type=['channel_sr', 'feat_sum'],
                 cnn_type='vgg16',
                 use_dcn=True,
                 use_cb=False,
                 use_bn=False,
                 nb_gaussian=8,
                 cnn_stride=16,
                 out_stride=8,
                 iosize=[480, 640, 60, 80],
                 time_dims=7,
                 cat_type=[0, 1, 0, 1],
                 pre_model_path=''):
        super(STRNN_SRFu_woAALSTM, self).__init__()

        self.time_dims = time_dims
        self.use_cb = use_cb
        self.nb_gaussian = nb_gaussian
        self.cat_type = cat_type

        # 1 Saliency Related Feature Extraction Module
        self.feat_sm = STRNN_Static_Net(cnn_type=cnn_type, use_dcn=use_dcn, use_cb=use_cb, use_bn=use_bn,
                                        time_dims=time_dims, cnn_stride=cnn_stride, nb_gaussian=nb_gaussian,
                                        pre_sf_path='')
        self.feat_of = STRNN_Dynamic_Net(out_stride=out_stride, use_flow=False, use_cb=use_cb,
                                         use_bn=use_bn, time_dims=time_dims, nb_gaussian=nb_gaussian,
                                         pre_of_path='')

        # 2 SR-Fu module, ratio = 16 same as SE-Block [70]
        # SE (CRM): pool_type = 'avg'; fusion_type = ['channel_mul']
        # GCM+CRM: pool_type = 'att' and fusion_type = ['channel_mul']
        # CRM+SRM: pool_type = 'avg' and fusion_type = ['channel_sr', 'feat_sum']
        # SR-Fu (GCM+CRM+SRM): pool_type = 'att'; fusion_type = ['channel_sr', 'feat_sum']
        expansion = numpy.sum(cat_type)
        self.att_channel = att_SR(256 * expansion, 256 * expansion // ratio, num_feat=expansion, pool=pool_type,
                                  fusions=fusion_type)

        self.feat_fu = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 3 AA-LSTM module, for ablation snalysis: change 'AttConvLSTM' --> 'ConvLSTM'
        # _, _, shape_r_out, shape_c_out = iosize
        # self.att_lstm = AttConvLSTM((shape_r_out, shape_c_out), 256, 256, 256, kernel_size=(3, 3), num_layers=1,
        #                             batch_first=True, bias=True, return_all_layers=False)

        self.conv_out = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
        )

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.load_state_dict(torch.load(pre_model_path).state_dict(), strict=False)

    def forward(self, x, cb, in_state=None):
        x_sm = normalize_data(x[:-1])
        x_of = x[:, [2, 1, 0], :, :]

        out_sf, x_sf, out_mf, x_mf = self.feat_sm(x_sm, cb)
        out_of, x_of, out_of_3d, x_of_3d = self.feat_of(x_of, cb)

        x_feat = [x_sf, x_mf, x_of, x_of_3d]
        x_cat = [x_feat[i] for i in range(4) if self.cat_type[i] == 1]
        x_fu = torch.cat(x_cat, 1)

        x_fu = self.att_channel(x_fu)
        x_fu = self.feat_fu(x_fu)

        # B_D, C, H, W = x_fu.size()
        # x_fu = x_fu.contiguous().view(1, B_D, C, H, W)
        # x_fu, x_state = self.att_lstm(x_fu, in_state)
        # x_fu = x_fu.contiguous().view(B_D, x_fu.size(2), H, W)

        out = F.relu(self.conv_out(x_fu))
        return out  # , x_state


class STRNN_simFu(nn.Module):
    def __init__(self, fu_type='sum',
                 cnn_type='vgg16',
                 use_dcn=True,
                 use_cb=False,
                 use_bn=False,
                 nb_gaussian=8,
                 cnn_stride=16,
                 out_stride=8,
                 time_dims=7,
                 cat_type=[0, 1, 0, 1],
                 pre_model_path=''):
        super(STRNN_simFu, self).__init__()

        assert fu_type in ['cat', 'sum']

        self.fu_type = fu_type
        self.time_dims = time_dims
        self.use_cb = use_cb
        self.nb_gaussian = nb_gaussian
        self.cat_type = cat_type

        # 1 Saliency Related Feature Extraction Module
        self.feat_sm = STRNN_Static_Net(cnn_type=cnn_type, use_dcn=use_dcn, use_cb=use_cb, use_bn=use_bn,
                                        time_dims=time_dims, cnn_stride=cnn_stride, nb_gaussian=nb_gaussian,
                                        pre_sf_path='')
        self.feat_of = STRNN_Dynamic_Net(out_stride=out_stride, use_flow=False, use_cb=use_cb,
                                         use_bn=use_bn, time_dims=time_dims, nb_gaussian=nb_gaussian,
                                         pre_of_path='')
        if fu_type == 'cat':
            expansion = numpy.sum(cat_type)
        else:
            expansion = 1

        self.feat_fu = nn.Sequential(
            nn.Conv2d(256 * expansion, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
        )

        init_weights(self.feat_fu)
        init_weights(self.conv_out)

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.load_state_dict(torch.load(pre_model_path).state_dict(), strict=False)

    def forward(self, x, cb, in_state=None):

        x_sm = normalize_data(x[:-1])
        x_of = x[:, [2, 1, 0], :, :]

        out_sf, x_sf, out_mf, x_mf = self.feat_sm(x_sm, cb)
        out_of, x_of, out_of_3d, x_of_3d = self.feat_of(x_of, cb)

        x_feat = [x_sf, x_mf, x_of, x_of_3d]
        x_cat = [x_feat[i] for i in range(4) if self.cat_type[i] == 1]
        x_fu = torch.cat(x_cat, 1)

        if self.fu_type == 'sum':
            b, c, h, w = x_fu.size()
            num_feat = len(x_cat)
            x_fu = x_fu.view(b, num_feat, c // num_feat, h, w)
            x_fu = torch.sum(x_fu, 1)

        x_fu = self.feat_fu(x_fu)
        out = F.relu(self.conv_out(x_fu))
        return out


class STRNN_SRFu_LSTM(nn.Module):
    def __init__(self, ratio=16,
                 pool_type='att',
                 fusion_type=['channel_sr', 'feat_sum'],
                 cnn_type='vgg16',
                 use_dcn=True,
                 use_cb=False,
                 use_bn=False,
                 nb_gaussian=8,
                 cnn_stride=16,
                 out_stride=8,
                 iosize=[480, 640, 60, 80],
                 time_dims=7,
                 cat_type=[0, 1, 0, 1],
                 pre_model_path=''):
        super(STRNN_SRFu_LSTM, self).__init__()

        self.time_dims = time_dims
        self.use_cb = use_cb
        self.nb_gaussian = nb_gaussian
        self.cat_type = cat_type

        # 1 Saliency Related Feature Extraction Module
        self.feat_sm = STRNN_Static_Net(cnn_type=cnn_type, use_dcn=use_dcn, use_cb=use_cb, use_bn=use_bn,
                                        time_dims=time_dims, cnn_stride=cnn_stride, nb_gaussian=nb_gaussian,
                                        pre_sf_path='')
        self.feat_of = STRNN_Dynamic_Net(out_stride=out_stride, use_flow=False, use_cb=use_cb,
                                         use_bn=use_bn, time_dims=time_dims, nb_gaussian=nb_gaussian,
                                         pre_of_path='')

        # 2 SR-Fu module, ratio = 16 same as SE-Block [70]
        # SE (CRM): pool_type = 'avg'; fusion_type = ['channel_mul']
        # GCM+CRM: pool_type = 'att' and fusion_type = ['channel_mul']
        # CRM+SRM: pool_type = 'avg' and fusion_type = ['channel_sr', 'feat_sum']
        # SR-Fu (GCM+CRM+SRM): pool_type = 'att'; fusion_type = ['channel_sr', 'feat_sum']
        expansion = numpy.sum(cat_type)
        self.att_channel = att_SR(256 * expansion, 256 * expansion // ratio, num_feat=expansion, pool=pool_type,
                                  fusions=fusion_type)

        self.feat_fu = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 3 AA-LSTM module, for ablation snalysis: change 'AttConvLSTM' --> 'ConvLSTM'
        _, _, shape_r_out, shape_c_out = iosize
        self.att_lstm = ConvLSTM((shape_r_out, shape_c_out), 256, 256, 256, kernel_size=(3, 3), num_layers=1,
                                 batch_first=True, bias=True, return_all_layers=False)

        self.conv_out = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
        )

        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.load_state_dict(torch.load(pre_model_path).state_dict(), strict=False)

    def forward(self, x, cb, in_state=None):
        x_sm = normalize_data(x[:-1])
        x_of = x[:, [2, 1, 0], :, :]

        out_sf, x_sf, out_mf, x_mf = self.feat_sm(x_sm, cb)
        out_of, x_of, out_of_3d, x_of_3d = self.feat_of(x_of, cb)

        x_feat = [x_sf, x_mf, x_of, x_of_3d]
        x_cat = [x_feat[i] for i in range(4) if self.cat_type[i] == 1]
        x_fu = torch.cat(x_cat, 1)

        x_fu = self.att_channel(x_fu)
        x_fu = self.feat_fu(x_fu)

        B_D, C, H, W = x_fu.size()
        x_fu = x_fu.contiguous().view(1, B_D, C, H, W)
        x_fu, x_state = self.att_lstm(x_fu, in_state)
        x_fu = x_fu.contiguous().view(B_D, x_fu.size(2), H, W)

        out = F.relu(self.conv_out(x_fu))
        return out, x_state


class STRNN_final_biAALSTM(nn.Module):
    def __init__(self, ratio=16,
                 pool_type='att',
                 fusion_type=['channel_sr', 'feat_sum'],
                 cnn_type='vgg16',
                 use_dcn=True,
                 use_cb=False,
                 use_bn=False,
                 nb_gaussian=8,
                 cnn_stride=16,
                 out_stride=8,
                 iosize=[480, 640, 60, 80],
                 time_dims=7,
                 cat_type=[1, 1, 1, 1],
                 bilstm_merge='sum',
                 pre_model_path=''):
        super(STRNN_final_biAALSTM, self).__init__()

        self.time_dims = time_dims
        self.use_cb = use_cb
        self.nb_gaussian = nb_gaussian
        self.cat_type = cat_type
        self.bilstm_merge = bilstm_merge

        # 1 Saliency Related Feature Extraction Module
        self.feat_sm = STRNN_Static_Net(cnn_type=cnn_type, use_dcn=use_dcn, use_cb=use_cb, use_bn=use_bn,
                                        time_dims=time_dims, cnn_stride=cnn_stride, nb_gaussian=nb_gaussian,
                                        pre_sf_path='')
        self.feat_of = STRNN_Dynamic_Net(out_stride=out_stride, use_flow=False, use_cb=use_cb,
                                         use_bn=use_bn, time_dims=time_dims, nb_gaussian=nb_gaussian,
                                         pre_of_path='')

        # 2 SR-Fu module, ratio = 16 same as SE-Block [70]
        # SE (CRM): pool_type = 'avg'; fusion_type = ['channel_mul']
        # GCM+CRM: pool_type = 'att' and fusion_type = ['channel_mul']
        # CRM+SRM: pool_type = 'avg' and fusion_type = ['channel_sr', 'feat_sum']
        # SR-Fu (GCM+CRM+SRM): pool_type = 'att'; fusion_type = ['channel_sr', 'feat_sum']
        expansion = numpy.sum(cat_type)
        self.att_channel = att_SR(256 * expansion, 256 * expansion // ratio, num_feat=expansion, pool=pool_type,
                                  fusions=fusion_type)

        self.feat_fu = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 3 AA-LSTM module, for ablation snalysis: change 'AttConvLSTM' --> 'ConvLSTM'
        _, _, shape_r_out, shape_c_out = iosize
        self.att_lstm = AttConvLSTM((shape_r_out, shape_c_out), 256, 256, 256, kernel_size=(3, 3), num_layers=1,
                                    batch_first=True, bias=True, return_all_layers=False)

        self.att_lstm_b = AttConvLSTM((shape_r_out, shape_c_out), 256, 256, 256, kernel_size=(3, 3), num_layers=1,
                                      batch_first=True, bias=True, return_all_layers=False)

        if self.bilstm_merge in ['cat']:
            last_in_channels = 512
        else:
            last_in_channels = 256

        self.conv_out = nn.Sequential(
            nn.Conv2d(last_in_channels, 1, kernel_size=3, padding=1),
        )

        # init_weights(self.feat_fu)
        # init_weights(self.conv_out)
        if os.path.exists(pre_model_path):
            print("Load pre-trained weights")
            self.load_state_dict(torch.load(pre_model_path).state_dict(), strict=False)

    def forward(self, x, cb, in_state=None):

        x_sm = normalize_data(x[:-1])
        x_of = x[:, [2, 1, 0], :, :]

        out_sf, x_sf, out_mf, x_mf = self.feat_sm(x_sm, cb)
        out_of, x_of, out_of_3d, x_of_3d = self.feat_of(x_of, cb)

        x_feat = [x_sf, x_mf, x_of, x_of_3d]
        x_cat = [x_feat[i] for i in range(4) if self.cat_type[i] == 1]
        x_fu = torch.cat(x_cat, 1)

        x_fu = self.att_channel(x_fu)
        x_fu = self.feat_fu(x_fu)

        B_D, C, H, W = x_fu.size()
        x_fu = x_fu.contiguous().view(1, B_D, C, H, W)

        x_fu_f = x_fu
        x_fu_f, x_state = self.att_lstm(x_fu_f, in_state)
        x_fu_f = x_fu_f.contiguous().view(B_D, x_fu_f.size(2), H, W)

        x_fu_b = x_fu.index_select(dim=1, index=torch.arange(B_D - 1, -1, -1).cuda())
        x_fu_b, _ = self.att_lstm_b(x_fu_b, None)
        x_fu_b = x_fu_b.contiguous().view(B_D, x_fu_b.size(2), H, W)

        if self.bilstm_merge in ['concat']:
            x_lstm = torch.cat([x_fu_b, x_fu_f], 1)
        else:
            x_lstm = x_fu_b + x_fu_f

        out = F.relu(self.conv_out(x_lstm))
        return out, x_state
