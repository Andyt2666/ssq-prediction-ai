#!/usr/bin/env python3
"""
src/model_tcn.py - 阶段 2: TCN 因果膨胀卷积网络与双头概率输出

架构特性:
1. 严格时间因果性: 因果卷积 (Causal Padding) 保证 t 时刻神经元仅依赖 <= t 历史
2. 膨胀卷积 (Dilated Conv): 感受野指数级扩展，覆盖更长历史走势
3. 双输出头设计:
   - 红球头 (Red Head): 33 维二值 Logits (BCEWithLogitsLoss 多标签损失)
   - 蓝球头 (Blue Head): 16 维多分类 Logits (CrossEntropyLoss)
4. 对照组: 包含基准 LSTM 模型 (LSTMModel) 以供对照实验
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """去除由于右侧 Padding 引入的未来信息，保证严格因果卷积"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SSQTCNModel(nn.Module):
    """双色球专属 TCN 双头网络"""
    def __init__(self, input_dim=91, num_channels=(64, 64, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        final_dim = num_channels[-1]

        # 双头分类器
        self.red_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 33)   # 33 个红球的独立 Logits
        )
        self.blue_head = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16)   # 16 类蓝球互斥分类 Logits
        )

    def forward(self, x):
        # x: [Batch, Time_Window, Feat_Dim] -> 转换为 Conv1d 需要的 [Batch, Feat_Dim, Time_Window]
        x_trans = x.transpose(1, 2)
        feat = self.tcn(x_trans)
        last_step = feat[:, :, -1]  # 提取最后一个时间步特征

        red_logits = self.red_head(last_step)
        blue_logits = self.blue_head(last_step)
        return red_logits, blue_logits


class SSQLSTMModel(nn.Module):
    """基线对比模型: 经典 LSTM 双头网络"""
    def __init__(self, input_dim=90, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.red_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 33)
        )
        self.blue_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_step = out[:, -1, :]
        red_logits = self.red_head(last_step)
        blue_logits = self.blue_head(last_step)
        return red_logits, blue_logits


def compute_loss(red_logits, blue_logits, target_red, target_blue, red_weight=1.0, blue_weight=0.5):
    """双任务综合损失函数"""
    red_loss = F.binary_cross_entropy_with_logits(red_logits, target_red)
    blue_loss = F.cross_entropy(blue_logits, target_blue)
    total_loss = red_weight * red_loss + blue_weight * blue_loss
    return total_loss, red_loss, blue_loss
