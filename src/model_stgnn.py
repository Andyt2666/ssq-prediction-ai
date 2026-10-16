#!/usr/bin/env python3
"""
src/model_stgnn.py - 复杂前沿架构: 时空图注意力与走势图视觉双流神经网络 (ST-GNN-Vision)

架构创新亮点:
1. 四重异质邻接矩阵 (4-fold Relational Graph Topology):
   - A_co: 历史统计共现图 (Empirical Co-occurrence)
   - A_trans: 跨期一阶马尔可夫转移图 (Markov Transition)
   - A_domain: 数论与彩票领域图 (连号邻接 + 同尾数连边 + 三区拓扑)
   - A_adaptive: 可学习自适应隐空间图 (Learnable End-to-End Latent Graph)
2. 走势图视觉双流分支 (Visual Trend Chart 2D Stream):
   - 将近 T 期的 [T, 33] 走势图视为 2D 点阵图像，通过 2D 卷积提取斜连号、头肩底、局部聚集等形态学特征
3. 时空图注意力 (Spatio-Temporal Graph Attention, ST-GAT):
   - 时间维: 因果膨胀门控时序单元 (Gated TCN Unit)
   - 空间维: 33 个红球节点在 4 种图拓扑上进行多头注意力消息传递
4. 跨模态注意力融合 (Cross-Modal Fusion):
   - 节点图表征与走势图全局视觉表征进行 Cross-Attention 交互，最终输出红蓝球概率
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_domain_adj_matrices(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据真实历史开奖构建数论图、共现图和转移图"""
    N = len(records)
    # 1. 数论与彩票领域图 A_domain
    A_domain = np.zeros((33, 33), dtype=np.float32)
    for i in range(33):
        for j in range(33):
            if i == j:
                A_domain[i, j] = 1.0
                continue
            # 连号关系 (|i - j| == 1)
            if abs(i - j) == 1:
                A_domain[i, j] += 0.8
            # 同尾号关系 (尾数相同, 如 03, 13, 23, 33)
            if (i + 1) % 10 == (j + 1) % 10:
                A_domain[i, j] += 0.6
            # 同区关系
            z_i = (i) // 11
            z_j = (j) // 11
            if z_i == z_j:
                A_domain[i, j] += 0.3

    # 2. 统计共现图 A_co
    co_occur = np.zeros((33, 33), dtype=np.float32)
    # 3. 马尔可夫转移图 A_trans (上一期开 i，下一期开 j)
    trans = np.zeros((33, 33), dtype=np.float32)

    prev_reds = None
    for r in records:
        reds = [int(r[f"r{k}"]) - 1 for k in range(1, 7)]
        # 统计共现
        for a in reds:
            for b in reds:
                co_occur[a, b] += 1.0

        # 统计转移
        if prev_reds is not None:
            for p in prev_reds:
                for c in reds:
                    trans[p, c] += 1.0
        prev_reds = reds

    # 归一化邻接矩阵 (对称归一化 D^{-0.5} A D^{-0.5})
    def normalize_adj(mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5, where=rowsum > 0)
        r_inv[rowsum == 0] = 0.0
        r_mat_inv = np.diag(r_inv.flatten())
        return r_mat_inv.dot(mx).dot(r_mat_inv)

    A_domain = normalize_adj(A_domain)
    A_co = normalize_adj(co_occur)
    # 转移矩阵使用行随机归一化
    rowsum_t = trans.sum(1, keepdims=True)
    rowsum_t[rowsum_t == 0] = 1.0
    A_trans = trans / rowsum_t

    return A_domain, A_co, A_trans


class MultiGraphGATLayer(nn.Module):
    """多图关系注意力卷积层 (Multi-Relational Graph Attention Layer)"""
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        # 4 种关系独立的线性变换权重
        self.fc_domain = nn.Linear(in_features, out_features, bias=False)
        self.fc_co = nn.Linear(in_features, out_features, bias=False)
        self.fc_trans = nn.Linear(in_features, out_features, bias=False)
        self.fc_adaptive = nn.Linear(in_features, out_features, bias=False)

        # 关系聚合门控
        self.relation_weights = nn.Parameter(torch.ones(4) / 4.0)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)
        self.relu = nn.GELU()

    def forward(self, x, adj_domain, adj_co, adj_trans, adj_adaptive):
        # x: [Batch, 33, in_features]
        # 对 4 种拓扑图分别进行图卷积消息聚合
        h_domain = torch.matmul(adj_domain, self.fc_domain(x))
        h_co = torch.matmul(adj_co, self.fc_co(x))
        h_trans = torch.matmul(adj_trans, self.fc_trans(x))
        h_adaptive = torch.matmul(adj_adaptive, self.fc_adaptive(x))

        # 软加权多图融合
        w = F.softmax(self.relation_weights, dim=0)
        h = w[0] * h_domain + w[1] * h_co + w[2] * h_trans + w[3] * h_adaptive

        out = self.norm(x + self.dropout(self.relu(h)))
        return out


class GatedTemporalUnit(nn.Module):
    """因果膨胀门控时序单元 (GLU 机制捕捉长程走势周期)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        # x: [Batch * 33, in_channels, Time]
        x_pad = F.pad(x, (self.padding, 0))  # 严格左侧因果填充
        f = torch.tanh(self.conv_f(x_pad))
        g = torch.sigmoid(self.conv_g(x_pad))
        return f * g  # 门控激活


class VisualTrendStream(nn.Module):
    """走势图 2D 视觉形态学表征分支 (Computer Vision Grid Stream)"""
    def __init__(self, out_dim=64):
        super().__init__()
        # 输入单通道走势图灰度图: [Batch, 1, Window_T, 33]
        self.conv_block = nn.Sequential(
            # 捕获局部斜连线 (kernel: 3x3)
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # 沿时间下采样，保留 33 列

            # 捕获长程跨行跨列走势 (kernel: 3x5)
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, trend_grid):
        # trend_grid: [Batch, 1, Window_T, 33]
        return self.conv_block(trend_grid)  # [Batch, out_dim]


class SSQComplexSTGNN(nn.Module):
    """
    复杂前沿模型: 多关系时空图注意力 + 走势图视觉双流网络
    """
    def __init__(
        self,
        node_in_feats=6,       # 每个红球节点的动态时序特征数 (命中标记, 遗漏值, 3个滚动频次, 冷热度)
        hidden_dim=64,
        num_layers=2,
        window_size=20,
        adj_domain=None,
        adj_co=None,
        adj_trans=None,
        dropout=0.2
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 固定的 3 个领域/统计邻接矩阵 (注册为 buffer，自动随模型转移到 GPU/CPU)
        self.register_buffer("adj_domain", torch.tensor(adj_domain, dtype=torch.float32) if adj_domain is not None else torch.eye(33))
        self.register_buffer("adj_co", torch.tensor(adj_co, dtype=torch.float32) if adj_co is not None else torch.eye(33))
        self.register_buffer("adj_trans", torch.tensor(adj_trans, dtype=torch.float32) if adj_trans is not None else torch.eye(33))

        # 可学习自适应图参数 (Learnable Latent Graph Embeddings)
        self.node_emb1 = nn.Parameter(torch.randn(33, 16) * 0.1)
        self.node_emb2 = nn.Parameter(torch.randn(33, 16) * 0.1)

        # 节点初始特征映射
        self.node_encoder = nn.Linear(node_in_feats, hidden_dim)

        # 时空图处理块 (ST-GNN Blocks)
        self.temporal_layers = nn.ModuleList([
            GatedTemporalUnit(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i)
            for i in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            MultiGraphGATLayer(hidden_dim, hidden_dim, num_heads=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 走势图 2D 视觉表征流
        self.vision_stream = VisualTrendStream(out_dim=hidden_dim)

        # 跨模态注意力融合层 (33 个节点分别与视觉全局上下文交互)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True, dropout=dropout)

        # 红球输出头 (每个节点输出一个独立激活 Logit)
        self.red_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)  # 33 个节点每个输出 1 维标量
        )

        # 蓝球输出头 (图全局注意力池化 + 蓝球全连接)
        self.blue_head = nn.Sequential(
            nn.Linear(hidden_dim + 16, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16)
        )

    def _get_adaptive_adj(self):
        """自适应图生成 A_adp = Softmax(ReLU(E1 @ E2^T))"""
        adp = F.relu(torch.matmul(self.node_emb1, self.node_emb2.t()))
        return F.softmax(adp, dim=-1)

    def forward(self, node_seq, trend_grid, blue_hist):
        """
        node_seq: [Batch, Window_T, 33, node_in_feats]  红球节点多维时序特征
        trend_grid: [Batch, 1, Window_T, 33]           走势图点阵图
        blue_hist: [Batch, 16]                         蓝球历史频次/独热特征
        """
        batch_size = node_seq.size(0)
        adj_adp = self._get_adaptive_adj()

        # 1. 节点特征初编码 [Batch, Window_T, 33, hidden_dim]
        h = self.node_encoder(node_seq)

        # 2. 时空图交替消息传递
        for t_layer, s_layer in zip(self.temporal_layers, self.spatial_layers):
            # 时间轴卷积: 展平节点 -> [Batch * 33, hidden_dim, Window_T]
            h_time = h.permute(0, 2, 3, 1).contiguous().view(batch_size * 33, self.hidden_dim, self.window_size)
            h_time = t_layer(h_time)
            # 还原形状 -> [Batch, Window_T, 33, hidden_dim]
            h = h_time.view(batch_size, 33, self.hidden_dim, self.window_size).permute(0, 3, 1, 2).contiguous()

            # 空间多图聚合: 针对最后一个时间步
            last_h = h[:, -1, :, :]  # [Batch, 33, hidden_dim]
            last_h = s_layer(last_h, self.adj_domain, self.adj_co, self.adj_trans, adj_adp)
            h = h.clone()
            h[:, -1, :, :] = last_h

        node_repr = h[:, -1, :, :]  # 最终各节点表征 [Batch, 33, hidden_dim]

        # 3. 走势图视觉表征提取
        vis_repr = self.vision_stream(trend_grid)  # [Batch, hidden_dim]
        vis_context = vis_repr.unsqueeze(1)        # [Batch, 1, hidden_dim]

        # 4. 跨模态注意力融合: 节点查询全局视觉走势形态
        fused_nodes, _ = self.cross_attn(query=node_repr, key=vis_context, value=vis_context)
        final_nodes = node_repr + fused_nodes  # [Batch, 33, hidden_dim]

        # 5. 红球头输出 33 维 Logits
        red_logits = self.red_head(final_nodes).squeeze(-1)  # [Batch, 33]

        # 6. 蓝球头输出 16 维 Logits (全局图池化 + 蓝球特征)
        graph_pooled = torch.mean(final_nodes, dim=1)        # [Batch, hidden_dim]
        blue_input = torch.cat([graph_pooled, blue_hist], dim=-1)
        blue_logits = self.blue_head(blue_input)             # [Batch, 16]

        return red_logits, blue_logits
