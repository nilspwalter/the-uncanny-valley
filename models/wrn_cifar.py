import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, conv_layer, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and conv_layer(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        #print("################")
        #print(x.shape)
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            if hasattr(self, "shape_list"):
                self.shape_list.append(x.shape)
        else:
            out = self.relu1(self.bn1(x))
            #print(out.shape)
            if hasattr(self, "shape_list"):
                self.shape_list.append(out.shape)

        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        #print(out.shape)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        #print(out.shape)
        if hasattr(self, "shape_list"):
            self.shape_list.append(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        #print("################")
        #print(self.equalInOut)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, conv_layer, stride, dropRate=0.0,
    name="standard"):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(
            conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate
        )
        self.name = name
    def _make_layer(
        self, conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    conv_layer,
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        conv_layer,
        linear_layer,
        depth=34,
        num_classes=10,
        widen_factor=10,
        dropRate=0.0,
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = conv_layer(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, conv_layer, 1, dropRate
        )
        # 1st sub-block
        self.sub_block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, conv_layer, 1, dropRate, "sub"
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, conv_layer, 2, dropRate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, conv_layer, 2, dropRate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = linear_layer(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, linear_layer):
                m.bias.data.zero_()

    def forward(self, x, ret_act=False):
        #print(x.shape)
        out = self.conv1(x)
        #print(out.shape)
        #print("Block1 start")
        out = self.block1(out)
        #print(out.shape)
        #print("Block2 start")
        out = self.block2(out)
        #print(out.shape)
        out = self.block3(out)
        #print(out.shape)
        out = self.relu(self.bn1(out))
        #print(out.shape)
        out = F.avg_pool2d(out, 8)
        #print(out.shape)
        out = out.view(-1, self.nChannels)
        #print(out.shape,"\n")
        if ret_act:
            return self.fc(out), out
        return self.fc(out)


# NOTE: Only supporting default (kaiming_init) initializaition.
def wrn_28_10(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=10, **kwargs)




def wrn_28_1(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=1, **kwargs)

def wrn_28_2(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=2, **kwargs)

def wrn_28_4(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=4, **kwargs)

def wrn_28_6(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=6, **kwargs)

def wrn_34_4(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=34, widen_factor=4, **kwargs)

def wrn_34_10(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=34, widen_factor=10, **kwargs)


def wrn_40_2(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=40, widen_factor=2, **kwargs)

