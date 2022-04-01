import torch
import torch.nn as nn
import torch.nn.functional as F
import os, math, numpy

init_func = {
    'uniform':nn.init.uniform_,
    'normal':nn.init.normal_,
    'constant':nn.init.constant_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_,
    'ones':nn.init.ones_,
    'zeros':nn.init.zeros_,
}


def init_weights(model, funcname='xavier_uniform', val=0.0):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if funcname == 'constant':
                init_func[funcname](m.weight,val)
            else:
                init_func[funcname](m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0.)

class att_SR(nn.Module):
    def __init__(self, inplanes, planes, num_feat=2, pool='att', fusions=['channel_sr', 'feat_sum','spatial_mul']):
        super(att_SR, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_mul', 'channel_sr', 'feat_sum','spatial_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.num_feat = num_feat
        self.pool = pool
        self.fusions = fusions

        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

        if 'channel_sr' in fusions:
            self.channel_sr_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_sr_conv = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            init_weights(self.conv_mask, 'kaiming_uniform')
        if self.channel_mul_conv is not None:
            init_weights(self.channel_mul_conv)
        if self.channel_sr_conv is not None:
            init_weights(self.channel_sr_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        spatial_att = torch.ones(batch, 1, height, width).cuda()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            spatial_att = context_mask.view(batch, 1, height, width)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context, spatial_att

    def forward(self, x):
        # [N, C, 1, 1]
        context, sp_att = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x

        if self.channel_sr_conv is not None:
            # [N, C, 1, 1]
            batch, channel, height, width = x.size()
            ch_att = self.channel_sr_conv(context).view(batch, self.num_feat, channel // self.num_feat)
            ch_att = torch.softmax(ch_att, 1)
            ch_att = ch_att.view(batch, channel, 1, 1)

            out = x * ch_att

        if 'spatial_mul' in self.fusions:
            out = out * sp_att

        if 'feat_sum' in self.fusions:
            out = out.view(batch, self.num_feat, channel // self.num_feat, height, width)
            out = torch.sum(out, 1)

        return out



