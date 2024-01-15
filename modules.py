import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
import torch.nn.functional as F

"""
use for 3CM
"""
class ChannelCorrection(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelCorrection, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights
class SpatialCorrection(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialCorrection, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())
        # add
        # self.conv = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights
class CoCorrenction(nn.Module):
    def __init__(self, dim, reduction=1, a=0.5, b=0.5):
        super(CoCorrenction, self).__init__()
        self.a = a
        self.b = b
        self.channel_c = ChannelCorrection(dim=dim, reduction=reduction)
        self.spatial_c = SpatialCorrection(dim=dim, reduction=reduction)

    def forward(self, x1, x2):
        channel_c = self.channel_c(x1, x2)
        spatial_c = self.spatial_c(x1, x2)
        out_x1 = x1 + self.a * channel_c[1] * x2 + self.b * spatial_c[1] * x2
        out_x2 = x2 + self.a * channel_c[0] * x1 + self.b * spatial_c[0] * x1
        return out_x1, out_x2

"""
use for SMFF
"""
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.channela = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.channela(x)
        out = torch.mul(x, att)
        return out
class SFeatureFusion(nn.Module):
    def __init__(self, in_channel, c=256, ratio=8):
        super(SFeatureFusion, self).__init__()
        self.g1 = GhostModule(in_channel, c)
        self.g2 = GhostModule(in_channel, c)
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv3 = nn.Conv2d(c, c, 3, 1, 1)
        self.ca1 = ChannelAttention(c, ratio)
        self.ca2 = ChannelAttention(c, ratio)

    def forward(self, rgb, depth, beta, gamma, gate):
        rgb = self.g1(rgb)
        depth = self.g2(depth)
        w1 = self.conv2(rgb)
        w2 = self.conv3(depth)
        feat_1 = F.relu(self.ca1(w2 * rgb), inplace=True)
        feat_2 = F.relu(self.ca2(w1 * depth), inplace=True)

        out1 = rgb + gate * beta * feat_1
        out2 = depth + (1.0 - gate) * gamma * feat_2

        return out1, out2



"""
use for DMFF
"""
class SA(nn.Module):
    def __init__(self, in_channel, out=256, norm_layer=nn.BatchNorm2d):
        super(SA, self).__init__()
        self.out = out
        self.conv1 = nn.Conv2d(in_channel, out, kernel_size=3, stride=1, padding=1)
        self.bn1 = norm_layer(out)
        self.conv2 = nn.Conv2d(out, out * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)
        w, b = out2[:, self.out:, :, :], out2[:, self.out:, :, :]
        return F.relu(w * out1 + b, inplace=True)
class CrossAttention(nn.Module):
    def __init__(self, in_channel=256, ratio=8):
        super(CrossAttention, self).__init__()
        self.conv_q = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1)
        self.conv_k = nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1)
        self.conv_v = nn.Conv2d(in_channel, in_channel, kernel_size=1)

    def forward(self, rgb, depth):
        bz, c, h, w = rgb.shape
        depth_q = self.conv_(depth).view(bz, -1, h * w).permute(0, 2, 1)
        depth_k = self.conv_k(depth).view(bz, -1, h * w)
        mask = torch.bmm(depth_q, depth_k)  # bz, hw, hw
        mask = torch.softmax(mask, dim=-1)
        rgb_v = self.conv_v(rgb).view(bz, c, -1)
        feat = torch.bmm(rgb_v, mask.permute(0, 2, 1))  # bz, c, hw
        feat = feat.view(bz, c, h, w)

        return feat
class DFeatureFusion(nn.Module):
    def __init__(self, in_channel, c=256, ratio=8):
        super(DFeatureFusion, self).__init__()
        self.sa1 = SA(in_channel,c)
        self.sa2 = SA(in_channel,c)
        self.att1 = CrossAttention(c, ratio=ratio)
        self.att2 = CrossAttention(c, ratio=ratio)

    def forward(self, rgb, depth, beta, gamma, gate):
        rgb = self.sa1(rgb)
        depth = self.sa2(depth)
        feat_1 = self.att1(rgb, depth)
        feat_2 = self.att2(depth, rgb)

        out1 = rgb + gate * beta * feat_1
        out2 = depth + (1.0 - gate) * gamma * feat_2

        return out1, out2



"""
use for weight summation
"""
class Convrelu_1(nn.Module):
    # convolution
    # relu
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1, bias=False):
        super(Convrelu_1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight.data)
    def forward(self, x):
        return F.relu(self.conv(x))
class Conv_3(nn.Module):
    # convolution
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super(Conv_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight.data)
    def forward(self, x):
        return self.conv(x)

"""
use for segformer encoder
"""
class DWConv(nn.Module):
    """
    Depthwise convolution bloc: input: x with size(B N C); output size (B N C)
    """
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous() # B N C -> B C N -> B C H W
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2) # B C H W -> B N C

        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        """
        MLP Block: 
        """
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Linear embedding
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # B N C -> B N num_head C//num_head -> B C//num_head N num_heads
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class Block(nn.Module):
    """
    Transformer Block: Self-Attention -> Mix FFN -> OverLap Patch Merging
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # B C H W
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        # B H*W/16 C
        x = self.norm(x)

        return x, H, W



"""
use for dffnet weighted summation
"""
class Fusion(nn.Module):
    def __init__(self, in_channel, out, norm_layer=nn.BatchNorm2d):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(64, out, 3, 1, 1)
        self.conv0 = nn.Conv2d(in_channel, out, 3, 1, 1)
        self.conv1 = nn.Conv2d(out*2, out, 3, 1, 1)
        self.bn0 = norm_layer(out)

    def forward(self, x1, x2, alpha):
        x1 = self.conv0(x1)
        x2 = self.conv0(x2)
        out1 = alpha * x1 +(1.0 - alpha) * x2
        out2 = x1 * x2
        out = torch.cat((out1, out2), dim=1)
        out = F.relu(self.bn0(self.conv1(out)), inplace=True)

        return out