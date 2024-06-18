from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from DMA import *

def get_inplanes():
    return [96, 192, 384, 768]
    # return [128, 256, 512, 1024]
    
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=7,
                     stride=stride,
                     padding=3,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, drop_path=0):
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=7, padding=3, stride=stride, groups=in_planes)
        self.pwconv1 = nn.Linear(planes, 4 * planes)
        self.conv2 = nn.Conv3d(4 * planes, 4 * planes, kernel_size=7, padding=3, stride=1, groups=4 * planes)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(4 * planes, planes)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = x.permute(0, 2, 3, 4, 1)
        x = nn.LayerNorm(x.size(), eps=1e-6).cuda()(x)
        x1 = self.pwconv1(x)
        x = x1.permute(0, 4, 1, 2, 3)
        x = self.conv2(x).permute(0, 2, 3, 4, 1)
        x = self.gelu(x) * x1
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3)

        return self.drop_path(x), residual


class MDCFANet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               bias=False)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.LayerNorm((self.in_planes, self.in_planes, self.in_planes), eps=1e-6),
                    nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=2, stride=stride)
                )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, weight): 
        x_1 = self.conv1(x)
        x_1 = nn.LayerNorm(x_1.size(), eps=1e-6).cuda()(x_1)
        if not self.no_max_pool:
            x = self.maxpool(x_1)

        x, residual = self.layer1(x)
        x, fmri = DMA(x.shape[1], 16)(x, weight)
        x = x + residual

        x, residual = self.layer2(x)
        x, fmri = DMA(x.shape[1], 16)(x, fmri + weight)
        x = x + residual

        x, residual = self.layer3(x)
        x, fmri = DMA(x.shape[1], 16)(x, fmri + weight)
        x = x + residual

        x, residual = self.layer4(x)
        x = x + residual

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(**kwargs):
    model = MDCFANet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    return model
