import numpy as np
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from build_vocab import WordVocab
import pandas as pd
import os
import torch.nn as nn
from utils import *
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool,global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential, Linear, ReLU
import torch
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
                                GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_add_pool,
                                Set2Set,
                                GCNConv,
                                )
#############################
class AttnConvBlock(nn.Module):
    def __init__(self, dim, heads=4, kernel_size=5, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 8* dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8* dim, dim)
        )

    def forward(self, x):  # x: [B, L, D]
        # Conv 分支（局部感受野）
        x_conv = self.conv(x.transpose(1, 2))  # [B, D, L]
        x_conv = x_conv.transpose(1, 2)  # [B, L, D]
        x_conv = self.norm1(x_conv)

        # Attention 分支（全局依赖）
        x_attn, _ = self.attn(x, x, x)  # [B, L, D]
        x_attn = self.norm2(x_attn)

        # 融合：concat + MLP
        x_cat = torch.cat([x_conv, x_attn], dim=-1)  # [B, L, 2D]
        out = self.mlp(x_cat)  # [B, L, D]
        return out

class SpatialGroupEnhance_for_1D(nn.Module):
    def __init__(self, groups = 32):
        super(SpatialGroupEnhance_for_1D, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1))
        self.sig      = nn.Sigmoid()
    
    def forward(self, x): # (b, c, h)
        b, c, h = x.size()
        x = x.reshape(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.reshape(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.reshape(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.reshape(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.reshape(b, c, h)
        return x

class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2) # (B,heads,seq_len)
        value = x # (B,seq_len,hidden_dim)

        minus_inf = -9e15 * torch.ones_like(query) # (B,heads,seq_len)
        e = torch.where(masks > 0.5, query, minus_inf)  # (B,heads,seq_len)
        a = self.softmax(e) # (B,heads,seq_len)

        out = torch.matmul(a, value) # (B,heads,seq_len) * (B,seq_len,hidden_dim) = (B,heads,hidden_dim)
        out = torch.mean(out, dim=1).squeeze() # (B,hidden_dim)
        return out, a
    
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=128, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        # target = data.target
        x = F.relu(self.conv1(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index)) # (B,seq_len,hidden_dim)
        x = self.bn5(x)
        x = global_add_pool(x, batch) # (B,hidden_dim)
        x = F.relu(self.fc1_xd(x)) # (B,hidden_dim)
        x = F.dropout(x, p=0.2, training=self.training) # (B,hidden_dim)
        # concat
        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class TCNBlock(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim //2, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(dim//2)
        self.relu = nn.ReLU()

    def forward(self, x):  # x: [B, L, D]
        x = x.transpose(1, 2)  # [B, D, L]
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out.transpose(1, 2) # [B, L, D//2]
class GlobalLocalAttention(nn.Module):
    def __init__(self, dim, window_size=128, dropout=0.2):
        super(GlobalLocalAttention, self).__init__()
        self.dim = dim
        self.window_size = dim
        self.scale = dim ** -0.5  # 简单的缩放因子

        # 用于生成 q, k, v 的线性层
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim*2)
        self.dropout = nn.Dropout(dropout)

        # Gating机制
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, N, D = x.shape  # 输入的形状是 (B, N, D)

        # 计算 q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, N, D) for t in qkv]  # (B, N, D)

        ### Local Attention ###
        pad = self.window_size // 2  # 计算 padding 大小
        k_padded = F.pad(k, (0, 0, pad, pad), mode='reflect')  # (B, N+2p, D)
        v_padded = F.pad(v, (0, 0, pad, pad), mode='reflect')  # (B, N+2p, D)

        k_local = k_padded.unfold(dimension=1, size=self.window_size, step=1)  # (B, N, W, D)
        v_local = v_padded.unfold(dimension=1, size=self.window_size, step=1)  # (B, N, W, D)
        k_local = k_local[:,:N,:,:]
        v_local = v_local[:,:N,:,:]
        q_exp = q.unsqueeze(2)  # (B, N, 1, D)
        attn_local = (q_exp * self.scale) @ k_local.transpose(-2, -1)  # (B, N, 1, W)
        attn_local = F.softmax(attn_local, dim=-1)

        out_local = attn_local @ v_local  # (B, N, 1, D)
        out_local = out_local.squeeze(2)  # (B, N, D)

        ### Global Attention ###
        attn_global = (q @ k.transpose(-2, -1)) * self.scale  # (B, N, N)
        attn_global = F.softmax(attn_global, dim=-1)
        out_global = attn_global @ v  # (B, N, D)

        # 最终输出
        # 拼接局部和全局特征来学习 gate（每个 token 一个门控）
        gate_input = torch.cat([out_local, out_global], dim=-1)  # (B, N, 2D)
        gate = self.gate_proj(gate_input)  # (B, N, 1)

        # 加权融合
        out = gate * out_local+ (1 - gate) * out_global  # (B, N, D)
        out = self.proj(self.dropout(out))

        return out

class CrossAttentionMultiHead(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttentionMultiHead, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"

        # Query, Key, Value 线性映射层
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)

        # 输出线性层
        self.out_linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        # 对 Q, K, V 进行线性映射
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # 将 Q, K, V 分成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 可选的掩码（比如用于遮挡未来位置的掩码）
        if mask is not None:
            attention_scores += mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        # 计算加权的值
        attention_output = torch.matmul(attention_probs, V)

        # 合并所有头的输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.head_dim)

        # 最终的线性变换
        output = self.out_linear(attention_output)
        output = output.squeeze(1)

        return output
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
class MultiLayerGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_layers):
        super(MultiLayerGCN, self).__init__()
        self.num_layers = num_layers
        # self.temp = 0.1
        # 创建多个 GCN 层
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))  # 第一层
        for _ in range(1, num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))  # 中间层
        self.convs.append(GCNConv(hidden_dim, num_classes))  # 最后一层

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)  # 图卷积
            x = F.relu(x)  # 激活函数
            x = F.dropout(x, training=self.training)  # Dropout

        x = self.convs[-1](x, edge_index)  # 最后一层卷积
        x = global_mean_pool(x, data.batch)
        x = F.softmax(x, dim=1)
        return x  # 使用 log_softmax 作为输出

import torch.nn.functional as F

class LMF(nn.Module):
    def __init__(self, dim, rank=4, dropout=0.1):
        """
        dim: 输入/输出的特征维度 D
        rank: LMF中的秩 r，控制融合能力与复杂度
        dropout: 融合后的dropout概率
        """
        super(LMF, self).__init__()
        self.rank = rank
        self.dim = dim

        # 每个模态的低秩张量因子 [r, D+1, D]
        self.factor1 = nn.Parameter(torch.Tensor(rank, dim + 1, dim))
        self.factor2 = nn.Parameter(torch.Tensor(rank, dim + 1, dim))
        self.factor3 = nn.Parameter(torch.Tensor(rank, dim + 1, dim))

        # 融合加权系数 [1, r] 和偏置 [1, D]
        self.fusion_weights = nn.Parameter(torch.Tensor(1, rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, dim))

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.factor1)
        nn.init.xavier_uniform_(self.factor2)
        nn.init.xavier_uniform_(self.factor3)
        nn.init.xavier_uniform_(self.fusion_weights)
        nn.init.zeros_(self.fusion_bias)

    def forward(self, x1, x2, x3):
        """
        x1, x2, x3: [B, D]
        return: [B, D]
        """
        B = x1.size(0)
        device = x1.device
        dtype = x1.dtype

        # Add bias term → [B, D+1]
        x1_ = torch.cat([torch.ones(B, 1, device=device, dtype=dtype), x1], dim=1)
        x2_ = torch.cat([torch.ones(B, 1, device=device, dtype=dtype), x2], dim=1)
        x3_ = torch.cat([torch.ones(B, 1, device=device, dtype=dtype), x3], dim=1)

        # 低秩投影：[B, r, D]
        proj1 = torch.einsum('bd, rdk -> brk', x1_, self.factor1)
        proj2 = torch.einsum('bd, rdk -> brk', x2_, self.factor2)
        proj3 = torch.einsum('bd, rdk -> brk', x3_, self.factor3)

        # 融合：[B, r, D]
        fused = proj1 * proj2 * proj3  # element-wise multiplication

        # 加权求和 rank 维度：[B, D]
        out = torch.einsum('br, brd -> bd', self.fusion_weights.expand(B, -1), fused) + self.fusion_bias
        out = self.dropout(out)
        return out
