import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm3d):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CostRegNet(nn.Module):
    def __init__(self, 
                 in_channels=32, 
                 light_weight=False,
                 norm_act=nn.InstanceNorm3d):
        super(CostRegNet, self).__init__()
        self.light_weight = light_weight
        # light_weight is used if the resolution of the cost volume is H/8 x W/8
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)
        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))
        self.depth_conv = nn.Sequential(nn.Conv3d(8, 1, 3, padding=1, bias=False))
        
    def add_up(self, x1, x2):
        # import ipdb; ipdb.set_trace()
        diffD = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4] 
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffD // 2, diffD - diffD // 2))
        if diffD !=0 or diffY != 0 or diffX != 0: x2 = F.interpolate(x2, size=x1.shape[-3:], mode='trilinear')
        return x1 + x2

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        if self.light_weight: x = conv4
        else: x = self.conv6(self.conv5(conv4)); x = conv4 + self.add_up(conv4, self.conv7(x))
        x = self.add_up(conv2, self.conv9(x))
        x = self.add_up(conv0, self.conv11(x))
        depth = self.depth_conv(x)
        return depth.squeeze(1)