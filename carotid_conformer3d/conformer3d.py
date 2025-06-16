import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from carotid_conformer3d.trans_encoder import trans_block
from collections import OrderedDict
from inflate_utils import ConvBlock_inflate, _inflate_conv_params, _inflate_bn_params

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module): # transformer block

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out = nn.Dropout(drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_out(self.attn(self.norm1(x)))
        # x = x + self.drop_out(self.mlp(self.norm2(x)))
        return x

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out = nn.Dropout(drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x = x + self.drop_out(self.attn(self.norm1(x)))
        # x = x + self.drop_out(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm3d, eps=1e-6), drop_block=None, drop_path=None,
                 inflate=None, inflate_pad=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion 
        
        # 1*1 Conv-BN
        self.conv1 = nn.Conv3d(inplanes, med_planes, kernel_size=(inflate[0], 1, 1), stride=(1, 1, 1), padding=(inflate_pad[0], 0, 0), bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)
        
        # 3*3 Conv-BN
        self.conv2 = nn.Conv3d(med_planes, med_planes, kernel_size=(inflate[1], 3, 3), stride=(1, stride, stride), groups=groups, padding=(inflate_pad[1], 1, 1), bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)
        
        # 1*1 Conv-BN
        self.conv3 = nn.Conv3d(med_planes, outplanes, kernel_size=(inflate[2], 1, 1), stride=(1, 1, 1), padding=(inflate_pad[2], 0, 0), bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)
        
        if res_conv:
            self.residual_conv = nn.Conv3d(inplanes, outplanes, kernel_size=(1, 1, 1), stride=(1, stride, stride), padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)
        
        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        self.outplanes = outplanes
        # 1*1 conv -> down module (avgpooling -> reshaping) -> layernorm
        self.conv_project = nn.Conv3d(inplanes, outplanes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0) # outplanes = 384 (align channel dim)
        self.sample_pooling = nn.AvgPool3d(kernel_size=(1, dw_stride, dw_stride), stride=(1, dw_stride, dw_stride))

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        N = x.shape[0]
        x = self.conv_project(x)  # [N, C, T, H, W]

        x = self.sample_pooling(x).flatten(3) # [N, 384, 8, 16, 16] -> [N, 384, 8, 256]
        x = x.transpose(2, 3) # [N, 384, 256, 8]
        x = x.contiguous().view(N, self.outplanes, -1).transpose(1, 2) # [N, 384, 256, 8] -> [N, 384, 2048] -> [N, 2048, 384]
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1) 

        return x

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, clip_len, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm3d, eps=1e-6)):
        super(FCUUp, self).__init__()
        # up module (reshaping -> interpolating)
        self.up_stride = up_stride
        self.clip_len = clip_len
        self.conv_project = nn.Conv3d(inplanes, outplanes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape

        # OUR TASK: [N, 2048+1, 384] -> [N, 2048, 384] -> [N, 384, 2048] -> [N, 384, 256, 8] -> [N, 384, 8, 256] -> [N, 384, 8, 16, 16] # patch size = 16
        x_r = x[:, 1:].transpose(1, 2).contiguous().reshape(B, C, -1, self.clip_len).transpose(2,3).reshape(B, C, self.clip_len, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(1, H * self.up_stride, W * self.up_stride)) # N, conv_block_channel(3*3 conv), outputsize


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, alpha, clip_len, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, 
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1, inflate=None, inflate_pad=None): # inplanes, outplanes = 256, 256

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups, inflate=inflate, inflate_pad=inflate_pad)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups, inflate=inflate, inflate_pad=inflate_pad)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups, inflate=inflate, inflate_pad=inflate_pad)
        
        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride) # conv block to trans block

        self.expand_block = FCUUp(clip_len, inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride) # trans block to conv block

        self.trans_block = trans_block(dim=embed_dim, num_heads=num_heads, num_frames=clip_len, alpha=alpha, mlp_ratio=mlp_ratio, 
                                           drop=drop_rate, attn_drop=attn_drop_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride) # N, C, T, H, W
        x = self.fusion_block(x, x_t_r, return_x_2=False) 

        return x, x_t


class Conformer3d(nn.Module):
    """
    Inflated conformer 2d architecture
    """

    def __init__(self, model_arch=None, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        
        super().__init__()

        # load inflate mode in all stages
        with open(model_arch) as f:
            model_info = yaml.load(f, Loader=yaml.FullLoader)
        
        self.inflate = model_info['backbone']
        self.num_classes = model_info['nc']
        self.clip_len = model_info['clip_len']
        self.embed_dim = embed_dim  # num_features for consistency with other models (default: 384)
        assert depth % 3 == 0 # The CNN branch and transformer branch are composed of N (depth) repeated convolution and transformer blocks, respectively

        # learnable parameter for fusion between cross dimension features
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.5)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.pooling = nn.AdaptiveAvgPool3d(1) 
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), embed_dim)
        self.cls_head = nn.Linear(embed_dim * 2, self.num_classes)

        # Stem stage: get the feature maps by conv block
        self.conv1 = nn.Conv3d(in_chans, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.act1 = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1, inflate=self.inflate[0][1], inflate_pad=self.inflate[0][2])
        # patch embedding
        self.trans_patch_conv = nn.Conv3d(64, embed_dim, kernel_size=(1, trans_dw_stride, trans_dw_stride), stride=(1, trans_dw_stride, trans_dw_stride), padding=0) 
        self.trans_1 = trans_block(dim=embed_dim, num_heads=num_heads, num_frames=self.clip_len, alpha=self.alpha, mlp_ratio=mlp_ratio, 
                                   drop=drop_rate, attn_drop=attn_drop_rate)
        

        # 2~4 stage (in c2 stage)
        # stage_1_channel = 256
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        self.alpha, self.clip_len, stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, inflate=self.inflate[1][1], inflate_pad=self.inflate[1][2]
                    )
            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage (in c3 stage)
        # stage_2_channel = 512
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        self.alpha, self.clip_len, in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, inflate=self.inflate[2][1], inflate_pad=self.inflate[2][2]
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage (in c4 & c5 stage)
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            if i == 12:
                pass
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        self.alpha, self.clip_len, in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion, inflate=self.inflate[3][1], 
                        inflate_pad=self.inflate[3][2]
                    )
            )
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)
        self.load_weights(pretrained_2d_path=model_info['pretrained_2d_path'], pretrained_path=model_info['pretrained_path'])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def load_weights(self, pretrained_2d_path=None, pretrained_path=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Default: None.
        """
        self.pretrained_2d_path = pretrained_2d_path # conformer2d pretrained weights
        self.pretrained_path = pretrained_path # conformer3d weights (ours)
        if isinstance(self.pretrained_2d_path, str):
            # Inflate 2D model into 3D model.
            print("Pretrained model: Inflate 2D model into 3D model")
            self._inflate_weights()

        elif isinstance(self.pretrained_path, str) and self.pretrained_2d_path is None: # Directly load 3D model.
            print("Pretrained model: Directly load 3D model")
            weights = torch.load(self.pretrained_path, map_location='cpu')['model']
            new_weights_dict = OrderedDict()
            if self.clip_len == 16:
                for key, v in weights.items():
                    new_key = key.split('module.')[-1]
                    new_weights_dict[new_key] = v
                self.load_state_dict(new_weights_dict, strict=True)
            else:
                self.load_state_dict(weights, strict=True)

        elif self.pretrained_2d_path is None and self.pretrained_path is None: # init weight
            print("Pretrained model: None (inited)")

        else:
            raise TypeError('pretrained must be a str or None')

    def _inflate_weights(self):
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            None.
        """

        state_dict_r2d = torch.load(self.pretrained_2d_path, map_location='cpu')

        inflated_param_names = [] # List of parameters that have been inflated.
        for name, module in self.named_children():
            if isinstance(module, ConvBlock):
                # ConvBlock: wrap 1*1+3*3+1*1conv + corresponding (bn+relu) + residual layers (exist in several blocks), thus the name mapping is needed
                ConvBlock_inflate(module, name, state_dict_r2d, inflated_param_names)
            elif isinstance(module, nn.Conv3d): # patch embedding conv (exclude conv1)
                original_conv_name = name
                if name != 'conv1': 
                    _inflate_conv_params(module, state_dict_r2d, original_conv_name, inflated_param_names)
            elif isinstance(module, nn.BatchNorm3d): # stem (bn1)
                original_bn_name = name 
                _inflate_bn_params(module, state_dict_r2d, original_bn_name, inflated_param_names)
            elif isinstance(module, ConvTransBlock):
                for ct_name, ct_module in module.named_children():
                    if isinstance(ct_module, ConvBlock):
                        ConvBlock_inflate(ct_module, f'{name}.{ct_name}', state_dict_r2d, inflated_param_names)
                    elif isinstance(ct_module, (FCUDown, FCUUp)):
                        original_conv_name = f'{name}.{ct_name}.conv_project'
                        _inflate_conv_params(ct_module.conv_project, state_dict_r2d, original_conv_name, inflated_param_names)
                        if hasattr(ct_module, 'bn'):
                            original_bn_name = f'{name}.{ct_name}.bn'
                            _inflate_bn_params(ct_module.bn, state_dict_r2d, original_bn_name, inflated_param_names)

        # parameters that can be loaded directly
        directly_load_dict = state_dict_r2d.copy()
        del_keys = ['conv_cls_head.weight', 'conv_cls_head.bias', 'conv1.weight', 'cls_head.weight', 'cls_head.bias',
                    'temporal relevant parameters of trans_1 or conv_trans_x']
        for key in list(directly_load_dict.keys()):
            if key in del_keys or key in inflated_param_names: # cannot be loaded (1. weights has been inflated 2. weights size mismatch)
                del directly_load_dict[key]
        
        # load the parameters that need not to be inflated in the inflate mode.
        deleted_keys, une_keys = self.load_state_dict(directly_load_dict, strict=False)
        print(f'These parameters in the 2d checkpoint are not loaded'
              f': {une_keys}')
        print(f'These parameters need to be initially weighted due to size mismatch or new layer'
              f': {del_keys}')
 
    def forward(self, x):
        B = x.shape[0]

        # stem stage [N, 1, 8, 256, 256] -> (conv1)[N, 64, 8, 128, 128] -> (max pooling)[N, 64, 8, 64, 64]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage (c2 stage: part 1)
        # [N, 256, 8, 64, 64]
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_patch_conv(x_base).flatten(3) # [N, 384, 8, 256]

        cls_tokens = self.cls_token.expand(B, -1, -1) # N, 1, 384
        x_t = x_t.transpose(2, 3) # [N, 384, 256, 8]
        x_t = x_t.contiguous().view(B, self.embed_dim, -1).transpose(1, 2) # [N, token_num(2048), 384]
        x_t = torch.cat([cls_tokens, x_t], dim=1) # [N, 2048+1, 384]
        
        x_t = self.trans_1(x_t) # [N, 2049, 384]
        
        # 2 ~ final (c2 stage: part 2 -> c5 stage)
        # final output: x -> [N, 1024, 8, 8] ; x_t -> [N, 257, 384]
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

        # conv classification
        x_p = self.pooling(x).flatten(1) # [N, 1024, 1, 1, 1] -> [N, 1024]
        conv_cls = self.conv_cls_head(x_p) # [N, 384]

        # trans classification
        x_t = self.trans_norm(x_t) # layernorm [N, 2049, 384]
        tran_cls = x_t[:, 0] # cls_token [N, 384]

        # concat
        pred = torch.cat((conv_cls, tran_cls), dim=-1) # [N, 768] for cls head

        # cls head
        result = self.cls_head(pred)
        return result



