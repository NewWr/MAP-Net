from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
from utils import *

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias).cuda()
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True).cuda() if bn else None
        self.relu = nn.GELU().cuda() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MA_C(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(MA_C, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.SiLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            ).cuda()
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale

class CPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class MA_S(nn.Module):
    def __init__(self):
        super(MA_S, self).__init__()
        self.compress = CPool()
    def forward(self, x, kernel_size):
        x_compress = self.compress(x)
        x_out = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class DMA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(DMA, self).__init__()
        self.MA_C = MA_C(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.MA_S = MA_S()

    def FST(self, weight, channel):
        input_dim = weight.shape[2]
        self.layer_n1 = nn.LayerNorm(input_dim).cuda()
        self.layer_n2 = nn.LayerNorm(input_dim).cuda()
        self.fc_1 = nn.Linear(input_dim, input_dim // 4).cuda()
        self.fc_2 = nn.Linear(input_dim // 4, input_dim).cuda()
        self.fc_3 = nn.Linear(weight.shape[1] * input_dim, channel).cuda()

        x1 = self.layer_n1(weight)
        x1_atten = MultiHeadAttention(n_head=8, d_k_=input_dim, d_v_=input_dim, d_k=input_dim, d_v=input_dim, d_o=input_dim).cuda()(x1, x1, x1)
        x = weight + x1_atten
        x1 = nn.ReLU()(self.fc_1(x))
        x11 = self.fc_2(x1)
        x11 = self.layer_n2(x11) + x
        x2 = self.fc_3(x11.reshape(x.shape[0], -1))

        return x2, x11

    def gate_mlp(self, x):
        x1 = nn.Linear(x.shape[1], x.shape[1] // 8).cuda()(x)
        x2 = nn.GELU().cuda()(x1)
        x = nn.Linear(x.shape[1] // 8, x.shape[1]).cuda()(x1 * x2)
        return x

    def avg_mlp(self, x, channel):
        a, b, c, d, e = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        size = c * d * e
        x_linear = x.reshape(a, b, -1)
        x1 = nn.Linear(size, size // 4).cuda()(x_linear)
        x1 = nn.Linear(size// 4, size).cuda()(x1)
        x1_act = nn.GELU().cuda()(x1).reshape(a, b, c, d, e)
        x1_avg = nn.AvgPool3d(4, 4).cuda()(x1_act)
        x1_max = nn.MaxPool3d(4, 4).cuda()(x1_act)
        x1_norm = nn.LayerNorm(x1_avg.size()).cuda()(x1_avg + x1_max).reshape(a, -1)
        channel_num = (c - c % 4) * (d - d % 4) * (e - e % 4) * b // 64
        x_out = nn.Linear(channel_num, channel).cuda()(x1_norm)
        return x_out

    def DMA(self, mri, fmri):
        a, b, c, d, e = mri.shape[0], mri.shape[1], mri.shape[2], mri.shape[3], mri.shape[4]
        conv_1 = nn.Conv3d(b, b, kernel_size=1, stride=1, groups=b).cuda()

        x = mri
        X1 = nn.Conv3d(b, b, kernel_size=3, stride=1, dilation=1, padding=1, groups=b).cuda()(x)
        X1 = nn.LayerNorm(X1.size()).cuda()(X1)
        X1 = F.gelu(X1)
        X2 = nn.Conv3d(b, b, kernel_size=11, stride=1, dilation=1, padding=5, groups=b).cuda()(x)
        X2 = nn.LayerNorm(X2.size()).cuda()(X2)
        X2 = F.gelu(X2)
        X3 = nn.Conv3d(b, b, kernel_size=7, stride=1, dilation=1, padding=3,  groups=b).cuda()(x)
        X3 = nn.LayerNorm(X3.size()).cuda()(X3)
        X3 = F.gelu(X3)

        X = X1 + X2 + X3
        X = conv_1(X)
        fmri_inf, fmri = self.FST(fmri, b)
        fmri_inf_cross = fmri_inf.view(a, b, 1)
        
        x1 = self.MA_C(X)
        
        x1_conv1 = MA_S()(x1, 3)
        x1_mlp_cross = self.avg_mlp(x1_conv1, b).view(a, b, 1)
        x1_crossatten = ScaledDotProductAttention(d_model=1, d_k=1, d_v=1, h=4).cuda()(x1_mlp_cross, fmri_inf_cross,
                                                                                       fmri_inf_cross)
        x1_crossatten = self.gate_mlp((x1_mlp_cross + F.sigmoid(x1_crossatten)
             * fmri_inf_cross).view(a, b)).view(a, b, 1, 1, 1)

        x1_channel = nn.BatchNorm3d(b).cuda()(x1_crossatten + x1_conv1)

        x2_conv1 = MA_S()(x1, 11)
        x2_mlp_cross = self.avg_mlp(x2_conv1, b).view(a, b, 1)
        x2_crossatten = ScaledDotProductAttention(d_model=1, d_k=1, d_v=1, h=4).cuda()(x2_mlp_cross, fmri_inf_cross,
                                                                                       fmri_inf_cross)
        x2_crossatten = self.gate_mlp(
            (x2_mlp_cross + F.sigmoid(x2_crossatten) * fmri_inf_cross).view(a, b)).view(a, b, 1, 1, 1)
        x2_channel = nn.BatchNorm3d(b).cuda()(x2_crossatten + x2_conv1)

        x3_conv1 = MA_S()(x1, 7)
        x3_mlp_cross = self.avg_mlp(x3_conv1, b).view(a, b, 1)
        x3_crossatten = ScaledDotProductAttention(d_model=1, d_k=1, d_v=1, h=4).cuda()(x3_mlp_cross, fmri_inf_cross,
                                                                                       fmri_inf_cross)
        x3_crossatten = self.gate_mlp(
            (x3_mlp_cross + F.sigmoid(x3_crossatten) * fmri_inf_cross).view(a, b)).view(a, b, 1, 1, 1)
        x3_channel = nn.BatchNorm3d(b).cuda()(x3_crossatten + x3_conv1)

        Y = x1_channel * X1 + x2_channel * X2 + x3_channel * X3

        return Y, fmri

    def forward(self, x, fmri):
        x_out, fmri = self.DMA(x, fmri)
        return x_out, fmri