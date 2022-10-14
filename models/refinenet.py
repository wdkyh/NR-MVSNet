import math

import torch
import torch.nn as nn
import torch.nn.functional as F

eps=1e-10


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.0,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2, dtype=torch.float) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen, dtype=torch.float).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size), dtype=torch.float)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, embedding):
        return self.dropout(embedding + self.pos_embedding[:embedding.size(0), :])


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FeatAttn(nn.Module):
    def __init__(self, in_channels, n_query_channels=32, patch_size=8,
                 embedding_dim=32, num_heads=4, num_layers=2, norm='linear'):
        super(FeatAttn, self).__init__()
        self.norm = norm
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.n_query_channels = n_query_channels
        
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.embedding_convPxP = nn.Conv2d(in_channels, self.embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = PositionalEncoding(self.embedding_dim)

        encoder_layers = nn.TransformerEncoderLayer(self.embedding_dim, num_heads, dim_feedforward=2 * self.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
    
    def forward(self, x):
        n, _, h, w = x.shape  # c = self.in_channels

        embeddings = self.embedding_convPxP(x).flatten(2)
        embeddings = embeddings.permute(2, 0, 1)
        embeddings = self.positional_encodings(embeddings)

        tgt = self.transformer_encoder(embeddings)

        queries = tgt[:self.n_query_channels, ...]
        
        feat = self.conv3x3(x)
        attn_feat = torch.matmul(feat.view(n, self.embedding_dim, h * w).permute(0, 2, 1), queries.permute(1, 2, 0))
        attn_feat = attn_feat.permute(0, 2, 1).view(n, self.n_query_channels, h, w)
        return attn_feat


class DepthUpdate(nn.Module):
    def __init__(self, feat_channels, n_query_channels):
        super(DepthUpdate, self).__init__()

        self.conv_cost = ConvBnReLU(
            in_channels=feat_channels, out_channels=n_query_channels, kernel_size=1, stride=1, pad=0
        )

        self.conv0 = ConvBnReLU(
            in_channels=n_query_channels + 1, out_channels=16, kernel_size=3, stride=1, pad=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.feat_attn_net = FeatAttn(feat_channels + 1, n_query_channels)

    def forward(self, ref_feat, cost_volume, prob_volume, depth, depth_interval_pixel, ndepth):
        '''
        ref_feat: [B, C, H, W]
        cost_volume: [B, C, D, H, W]
        prob_volume: [B, D, H, W]
        depth: [B, 1, H, W]
        depth_interval_pixel: float
        ndepth: int
        '''
        depth_min = (depth - ndepth / 2 * depth_interval_pixel)  # (B, H, W)
        depth_max = (depth + ndepth / 2 * depth_interval_pixel)
        depth = (depth - depth_min) / (depth_max - depth_min + eps)

        # reference featrue attention
        ref_feat_attn = self.feat_attn_net(torch.cat([ref_feat, depth], dim=1))

        # cost volume attention
        with torch.no_grad():
            entropy = torch.div(
                torch.sum(prob_volume * prob_volume.clamp(1e-9, 1.).log(), dim=1, keepdim=True),
                -math.log(ndepth)
            )
            mask = (1 - entropy) > 0.3
        cost_feat_attn = torch.sum(cost_volume * prob_volume.unsqueeze(1), dim=2) * mask.to(torch.float32)
        cost_feat_attn = self.conv_cost(cost_feat_attn)

        # update depth
        update_input = torch.cat([cost_feat_attn + ref_feat_attn, depth], dim=1)
        depth_res = self.conv2(self.conv1(self.conv0(update_input)))
        new_depth = (depth + depth_res) * (depth_max - depth_min) + depth_min

        return new_depth.squeeze(1)


# class DepthUpdate(nn.Module):
#     def __init__(self, in_channels_x1, in_channels_x2):
#         super(DepthUpdate, self).__init__()

#         self.conv1x1 = ConvBnReLU(in_channels_x2, out_channels=in_channels_x1, kernel_size=1, stride=1, pad=0)

#         in_channels = in_channels_x1 * 2
#         self.conv0 = ConvBnReLU(in_channels, out_channels=16, kernel_size=3, stride=1, pad=1)
#         self.conv1 = ConvBnReLU(in_channels=16, out_channels=8, kernel_size=3, stride=1, pad=1)

#         self.conv2 = ConvBnReLU(in_channels=1, out_channels=8, kernel_size=3, stride=1, pad=1)

#         self.conv3 = ConvBnReLU(in_channels=16, out_channels=8, kernel_size=3, stride=1, pad=1)
#         self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

#     def forward(self, x1, x2, depth, depth_interval_pixel, ndepth):

#         depth_min = (depth - ndepth / 2 * depth_interval_pixel)  # (B, H, W)
#         depth_max = (depth + ndepth / 2 * depth_interval_pixel)
#         feat1 = self.conv1x1(x2)
#         feat1 = torch.cat((feat1, x1), dim=1)
#         feat1 = self.conv1(self.conv0(feat1))
#         # print(depth_min.shape, depth_min.dim)
#         if depth_min.dim() == 2:
#             depth_min = depth_min.unsqueeze(2).unsqueeze(3)
#             depth_max = depth_max.unsqueeze(2).unsqueeze(3)
#         depth = (depth - depth_min) / (depth_max - depth_min + eps)
#         feat2 = self.conv2(depth)

#         depth_res = self.conv4(self.conv3(torch.cat((feat1, feat2), dim=1)))
#         new_depth = depth + depth_res
#         new_depth = new_depth * (depth_max - depth_min) + depth_min
        
#         return new_depth.squeeze(1)