#!/usr/bin/env python3
"""
src/dataset_stgnn.py - 时空图与走势图多模态数据集构建器

将历史开奖转化为:
1. 33 个节点的局部时序动态特征: [T, 33, 6]
2. 走势图 2D 点阵视觉张量: [1, T, 33] (类似黑白图像)
3. 蓝球历史分布张量: [16]
"""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class SSQGraphVisualDataset(Dataset):
    def __init__(self, records: list[dict], window_size: int = 20):
        self.records = records
        self.n_issues = len(records)
        self.window_size = window_size
        self.length = self.n_issues - self.window_size

        # 构建全时段基础矩阵
        self.red_grid = np.zeros((self.n_issues, 33), dtype=np.float32)
        self.blue_labels = np.zeros(self.n_issues, dtype=np.int64)

        for idx, r in enumerate(records):
            reds = [int(r[f"r{k}"]) - 1 for k in range(1, 7)]
            self.red_grid[idx, reds] = 1.0
            self.blue_labels[idx] = int(r["blue"]) - 1

        # 逐期预计算 33 个节点的 6 维高级时序特征
        # [0]: 当期命中 0/1
        # [1]: 遗漏期数 (归一化)
        # [2]: 近5期频次
        # [3]: 近10期频次
        # [4]: 近30期频次
        # [5]: 连续冷热温分值
        self.node_features = np.zeros((self.n_issues, 33, 6), dtype=np.float32)
        current_gaps = np.zeros(33, dtype=np.float32)

        for t in range(self.n_issues):
            self.node_features[t, :, 0] = self.red_grid[t]
            self.node_features[t, :, 1] = current_gaps / 25.0

            # 滚动频次 (严格不包含未来)
            w5_start = max(0, t - 5)
            w10_start = max(0, t - 10)
            w30_start = max(0, t - 30)

            self.node_features[t, :, 2] = np.mean(self.red_grid[w5_start:t+1], axis=0) if t > 0 else 0
            self.node_features[t, :, 3] = np.mean(self.red_grid[w10_start:t+1], axis=0) if t > 0 else 0
            self.node_features[t, :, 4] = np.mean(self.red_grid[w30_start:t+1], axis=0) if t > 0 else 0
            # 冷热态得分 (近期频次减去理论期望 6/33 ≈ 0.1818)
            self.node_features[t, :, 5] = (self.node_features[t, :, 3] - 0.1818) * 5.0

            # 更新遗漏
            for num in range(33):
                if self.red_grid[t, num] == 1.0:
                    current_gaps[num] = 0.0
                else:
                    current_gaps[num] += 1.0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 窗口 [idx, idx + window_size)
        target_t = idx + self.window_size

        # 1. 节点多维时序张量 [T, 33, 6]
        node_seq = self.node_features[idx: target_t]

        # 2. 走势图 2D 图像张量 [1, T, 33]
        trend_grid = self.red_grid[idx: target_t][np.newaxis, :, :]

        # 3. 蓝球历史统计特征 [16]
        blue_window = self.blue_labels[idx: target_t]
        blue_hist = np.zeros(16, dtype=np.float32)
        for b in blue_window:
            blue_hist[b] += 1.0
        blue_hist /= max(1.0, float(self.window_size))

        # 目标标签
        y_red = self.red_grid[target_t]
        y_blue = self.blue_labels[target_t]

        return (
            torch.tensor(node_seq, dtype=torch.float32),
            torch.tensor(trend_grid, dtype=torch.float32),
            torch.tensor(blue_hist, dtype=torch.float32),
            torch.tensor(y_red, dtype=torch.float32),
            torch.tensor(y_blue, dtype=torch.long)
        )
