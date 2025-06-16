"""
transformer block built with divided space-time self attention (T+S)
"""
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange

class TemporalAttention(nn.Module):
    """Temporal Attention in Parallel Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to 0.1.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`. Defaults to 0.1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query):
        """
        Args:
            query: shape as [N, 1+(256*8), 384]
        """
        identity = query.clone()

        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_t = query[:, 1:, :]

        b, pt, m = query_t.size() 
        p, t = pt // self.num_frames, self.num_frames 

        cls_token = init_cls_token.repeat(1, p, 1).reshape(b * p, m).unsqueeze(1)

        query_t = query_t.reshape(b * p, t, m) 
        query_t = torch.cat((cls_token, query_t), 1)
        query_t = query_t.permute(1, 0, 2) 
        
        res_temporal = self.attn(query_t, query_t, query_t)[0].permute(1, 0, 2)
        res_temporal = self.proj_drop(res_temporal.contiguous())

        cls_token = res_temporal[:, 0, :].reshape(b, p, m) 
        cls_token = torch.mean(cls_token, 1, True)

        res_temporal = res_temporal[:, 1:, :].reshape(b, p * t, m)
        res_temporal = torch.cat((cls_token, res_temporal), 1)

        new_query = identity + res_temporal
        return new_query

class SpatialAttention(nn.Module):
    """Spatial Attention in Parallel Space Time Attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder.
        num_frames (int): Number of frames in the video.
        attn_drop (float): A Dropout layer on attn_output_weights. Defaults to 0.1.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`. Defaults to 0.1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query):
        identity = query.clone()
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]

        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames

        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t, m).unsqueeze(1)

        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)
        query_s = query_s.permute(1, 0, 2)
        
        res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        res_spatial = self.proj_drop(res_spatial.contiguous())

        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
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

class trans_block(nn.Module): # transformer block built with divided space-time self attention (T+S)

    def __init__(self, dim, num_heads, num_frames, alpha, mlp_ratio=4., drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm_temporal = norm_layer(dim)
        self.norm_spatial = norm_layer(dim)
        self.attn_temporal = TemporalAttention(dim,
                                               num_heads,
                                               num_frames,
                                               attn_drop,
                                               proj_drop=drop)
        self.attn_spatial = SpatialAttention(dim,
                                             num_heads,
                                             num_frames,
                                             attn_drop,
                                             proj_drop=drop)
        self.norm_ffn = norm_layer(dim) # before mlp
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # learnable parameter for fusion between cross dimension features
        self.alpha = alpha

    def forward(self, x):
        """
        Args:
            x: shape as [N, 1+(256*8), 384]
        Return:
            same shape as input
        """
        # spatial MSA (including skip connection)
        x1 = self.attn_spatial(self.norm_spatial(x))

        # temporal MSA (including skip connection)
        x2 = self.attn_temporal(self.norm_temporal(x))
        
        # adaptive fusion
        x = self.alpha * x1 + (1 - self.alpha) * x2
        
        # FFN with norm
        x = x + self.mlp(self.norm_ffn(x))
        
        return x # [N, 1+(256*8), 384]

