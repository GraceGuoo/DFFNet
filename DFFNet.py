import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial

from modules import Convrelu_1, Conv_3, Fusion
from toolbox.losses import lovasz_softmax

from engine.logger import get_logger
logger = get_logger()

class DFFNet(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(weight = torch.from_numpy(np.array([1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()), norm_layer=nn.BatchNorm2d):
        super(DFFNet, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer

       # segformer encoder
        if cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from encoder import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from encoder import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from encoder import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from encoder import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        # mlpdecoder
        from MLPDecoder import MLPDecoderHead
        self.rgb_head = MLPDecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        self.rgbx_head = MLPDecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)

        self.linear_pred = nn.Conv2d(cfg.decoder_embed_dim, cfg.num_classes, kernel_size=1)
        self.linear_pred_out = nn.Conv2d(256, cfg.num_classes, kernel_size=1)

        self.criterion = criterion
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

        # weighted summation
        self.d_weight_classifier_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.d_weight_classifier = nn.Sequential(
            Convrelu_1(512, 256),
            Conv_3(256, 256),
            nn.Sigmoid()
        )
        self.fusion = Fusion(cfg.decoder_embed_dim, 256)


    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        orisize = rgb.shape
        x_rgb, x_e, gate = self.backbone(rgb, modal_x)

        out1 = self.rgb_head.forward(x_rgb)
        out2 = self.rgbx_head.forward(x_e)

        d = self.d_weight_classifier_avgpool( torch.cat((out1, out2), dim=1))
        d = self.d_weight_classifier(d)
        fuse = self.fusion(out1, out2, d)

        out = F.interpolate(self.linear_pred_out(fuse), size=orisize[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.linear_pred(out1), size=orisize[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.linear_pred(out2), size=orisize[2:], mode='bilinear', align_corners=False)

        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm

        return out, out1, out2

    def forward(self, rgb, modal_x, label=None):

        out, out1, out2 = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss1 = self.criterion(out, label.long())
            # loss1_1 = lovasz_softmax(F.softmax(out, dim=1), label.long(), ignore=255)
            loss2 = self.criterion(out1, label.long())
            loss3 = self.criterion(out2, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss1  + loss2 + loss3 #+ loss1_1

        return out, out1, out2