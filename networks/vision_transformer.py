# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn

from .swin_transformer_unet_skip_expand_decoder import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, config):
        super(SwinUnet, self).__init__()
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            weights = torch.load(pretrained_path, map_location=device)
            model_dict = self.swin_unet.state_dict()
            if "model" not in weights:
                print("---start load pretrained model---")
                weights = {k[17:]: v for k, v in weights.items()}
                for k in list(weights.keys()):
                    if k not in model_dict:
                        print("delete key:{}".format(k))
                        del weights[k]
                        continue
                    if weights[k].shape != model_dict[k].shape:
                        v = weights[k]
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del weights[k]
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del weights[k]

                self.swin_unet.load_state_dict(weights, strict=False)

                return
            pretrained_dict = weights['model']
            print("---start load pretrained model---")
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if k.startswith('swin_unet.'):
                    current_k = k[10:]
                    full_dict.update({current_k: v})
                    del full_dict[k]
                elif 'module.swin_unet.' in k:
                    current_k = k[17:]
                    full_dict.update({current_k: v})
                    del full_dict[k]
                else:
                    if "layers." in k:
                        current_layer_num = 3 - int(k[7:8])
                        current_k = "layers_up." + str(current_layer_num) + k[8:]
                        full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")
