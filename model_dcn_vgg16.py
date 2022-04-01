import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import numpy as np


__all__ = [
    'VGG', 'vgg16', 'vgg16_bn',
]


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, layers,
                 batch_norm=False,
                 strides=[2, 2, 2, 1],
                 dilations=[1, 1, 1, 1],
                 MG_rates=[1, 1, 1],
                 init_weights=True):
        super(VGG, self).__init__()

        if not strides :
            strides = [2, 2, 2, 1]
        if not dilations:
            dilations = [1, 1, 1, 1]

        self.inplanes = 3
        self.layer1 = self._make_layers(layers[0], 64,  batch_norm, pool_stride=2, dilation=1)
        self.layer2 = self._make_layers(layers[1], 128, batch_norm, pool_stride=2, dilation=dilations[0])
        self.layer3 = self._make_layers(layers[2], 256, batch_norm, pool_stride=2, dilation=dilations[1])
        self.layer4 = self._make_layers(layers[3], 512, batch_norm, pool_stride=strides[2], dilation=dilations[2])
        self.layer5 = self._make_layers(layers[4], 512, batch_norm, pool_stride=strides[3], dilation=dilations[3], pool=False)
        # self.layer5 = self._make_layers(layers[4], 512, batch_norm, pool_stride=strides[3], dilation=dilations[3])
        # self.layer5 = self._make_MG_layers(layers[4], 512, batch_norm, pool=False, pool_stride=strides[3], dilation=dilations[3], MG_rates=MG_rates)

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 1000),
        # )
        if init_weights:
            self._initialize_weights()

    def _make_layers(self, blocks, planes, batch_norm=False, pool=True, pool_stride=2, dilation=1):
        layers = []
        for i in range(blocks):
            conv2d = nn.Conv2d(self.inplanes, planes, kernel_size=3, padding=dilation, dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            self.inplanes = planes
        if pool:
            layers += [nn.MaxPool2d(kernel_size=2, stride=pool_stride)]
        return nn.Sequential(*layers)

    def _make_MG_layers(self, blocks, planes, batch_norm=False, pool=True, pool_stride=2, dilation=1, MG_rates=[1,1,1]):

        num_mg = len(MG_rates)
        if num_mg < blocks:
            resize_mg = np.ones((blocks))
            resize_mg[blocks - num_mg:] = MG_rates
        else:
            resize_mg = MG_rates[blocks - num_mg:]

        layers = []
        for i in range(blocks):
            conv2d = nn.Conv2d(self.inplanes, planes, kernel_size=3, padding=dilation*resize_mg[i], dilation=dilation*resize_mg[i])
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            self.inplanes = planes
        if pool:
            layers += [nn.MaxPool2d(kernel_size=2, stride=pool_stride)]
        return nn.Sequential(*layers)

    def forward(self, x):

        s1 = self.layer1(x)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        s5 = self.layer5(s4)

        return s1, s2, s3, s4, s5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def load_weight(model, model_urls):

    premodel_dic = model_zoo.load_url(model_urls)
    model_dic = model.state_dict()

    if len(premodel_dic) != len(model_dic):
        raise NotImplementedError

    tmp_dic = {k: v for k, v in zip(model_dic.keys(),premodel_dic.values())}
    model_dic.update(tmp_dic)
    model.load_state_dict(model_dic)
    return model


def vgg16(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG([2,2,3,3,3], batch_norm=False, **kwargs)
    if pretrained:
        # load_weight(model, model_urls['vgg16'])
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']),False)
    return model


def vgg16_bn(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG([2,2,3,3,3], batch_norm=True, **kwargs)
    if pretrained:
        load_weight(model, model_urls['vgg16_bn'])
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model




if __name__ == '__main__':
    print("test vgg")

    from torchvision import transforms
    img_transform = transforms.Compose([
        transforms.Resize((480,640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    import PIL.Image as Img
    img = Img.open("E:/COCO_train2014_000000000009.jpg").convert('RGB')
    img = img_transform(img).unsqueeze(0)

    from torchsummary import summary
    model = vgg16(True).cuda()

    summary(model,(3,480,640))

    model.eval()
    out = model(img.cuda())
    for i in range(len(out)):
        print(out[i].shape)

    import matplotlib.pyplot as plt
    import numpy as np
    plt.imshow(out.data.cpu().numpy())
    plt.show()
