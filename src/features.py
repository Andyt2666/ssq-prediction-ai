#!/usr/bin/env python3
"""
src/features.py - 阶段 1: 彩票统计特征工程与时序张量构建

功能:
1. 基础特征: 33 维红球多热编码 (Multi-Hot)、16 维蓝球独热编码 (One-Hot)
2. 统计特征: 遗漏值矩阵 (Omission Gap)、近 N 期出号频次、滚动和值、奇偶比、大小比、连号数、三区分布、AC 值
3. 时序数据集构建: [N, T, C] 张量，支持严格时间序列切分 (无未来信息泄露)
"""

import csv
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def compute_ac_value(reds: list[int]) -> int:
    """计算红球 AC 值 (数字复杂度): 任意两两差值的独立数量 - (6 - 1)"""
    s_reds = sorted(reds)
    diffs = set()
    for i in range(len(s_reds)):
        for j in range(i + 1, len(s_reds)):
            diffs.add(s_reds[j] - s_reds[i])
    return len(diffs) - (len(s_reds) - 1)


def count_consecutive(reds: list[int]) -> int:
    """计算红球最大连号长度 (如 14, 15 为 2 连号)"""
    s_reds = sorted(reds)
    max_c = 1
    cur_c = 1
    for i in range(1, len(s_reds)):
        if s_reds[i] == s_reds[i - 1] + 1:
            cur_c += 1
            max_c = max(max_c, cur_c)
        else:
            cur_c = 1
    return max_c


class SSQFeatureExtractor:
    """双色球全维度特征提取器"""

    def __init__(self, records: list[dict]):
        # records 保证按时间从旧到新排列
        self.records = records
        self.n_issues = len(records)
        self.red_multi_hot = np.zeros((self.n_issues, 33), dtype=np.float32)
        self.blue_one_hot = np.zeros((self.n_issues, 16), dtype=np.float32)
        self.blue_labels = np.zeros((self.n_issues,), dtype=np.int64)

        # 逐期统计特征
        self.sum_red = np.zeros((self.n_issues, 1), dtype=np.float32)
        self.span_red = np.zeros((self.n_issues, 1), dtype=np.float32)
        self.odd_ratio = np.zeros((self.n_issues, 1), dtype=np.float32)
        self.high_ratio = np.zeros((self.n_issues, 1), dtype=np.float32)
        self.ac_values = np.zeros((self.n_issues, 1), dtype=np.float32)
        self.consecutive = np.zeros((self.n_issues, 1), dtype=np.float32)
        self.zones = np.zeros((self.n_issues, 3), dtype=np.float32)

        self._parse_base()
        self._compute_dynamic_omissions()

    def _parse_base(self):
        for idx, r in enumerate(self.records):
            reds = [int(r[f"r{i}"]) for i in range(1, 7)]
            blue = int(r["blue"])

            # 独热/多热 (1-indexed -> 0-indexed)
            for num in reds:
                self.red_multi_hot[idx, num - 1] = 1.0
            self.blue_one_hot[idx, blue - 1] = 1.0
            self.blue_labels[idx] = blue - 1

            # 统计指标
            self.sum_red[idx, 0] = sum(reds) / 100.0  # 归一化至 ~1.0
            self.span_red[idx, 0] = (reds[5] - reds[0]) / 32.0
            self.odd_ratio[idx, 0] = sum(1 for x in reds if x % 2 == 1) / 6.0
            self.high_ratio[idx, 0] = sum(1 for x in reds if x > 16) / 6.0
            self.ac_values[idx, 0] = compute_ac_value(reds) / 10.0
            self.consecutive[idx, 0] = count_consecutive(reds) / 6.0

            # 三区分布: 1-11, 12-22, 23-33
            z1 = sum(1 for x in reds if 1 <= x <= 11) / 6.0
            z2 = sum(1 for x in reds if 12 <= x <= 22) / 6.0
            z3 = sum(1 for x in reds if 23 <= x <= 33) / 6.0
            self.zones[idx] = [z1, z2, z3]

    def _compute_dynamic_omissions(self):
        """计算逐期实时遗漏矩阵: 杜绝未来信息，严格只用当前期及以前"""
        self.red_omissions = np.zeros((self.n_issues, 33), dtype=np.float32)
        cur_gap = np.zeros(33, dtype=np.float32)

        for idx in range(self.n_issues):
            self.red_omissions[idx] = cur_gap / 20.0  # 归一化缩放
            # 更新当期
            for num_idx in range(33):
                if self.red_multi_hot[idx, num_idx] == 1.0:
                    cur_gap[num_idx] = 0.0
                else:
                    cur_gap[num_idx] += 1.0

    def get_feature_matrix(self) -> np.ndarray:
        """拼接成单期的全部特征向量:
        33 (红球) + 16 (蓝球) + 33 (遗漏) + 1(和值) + 1(跨度) + 1(奇偶) + 1(大小) + 1(AC) + 1(连号) + 3(三区)
        总维度 = 33 + 16 + 33 + 8 = 90
        """
        all_feats = np.concatenate([
            self.red_multi_hot,
            self.blue_one_hot,
            self.red_omissions,
            self.sum_red,
            self.span_red,
            self.odd_ratio,
            self.high_ratio,
            self.ac_values,
            self.consecutive,
            self.zones
        ], axis=1)
        return all_feats


class SSQDataset(Dataset):
    """时序滑动窗口数据集 (严格无前视泄漏)"""

    def __init__(self, feature_matrix: np.ndarray, red_labels: np.ndarray, blue_labels: np.ndarray, window_size: int = 20):
        self.features = feature_matrix
        self.red_labels = red_labels
        self.blue_labels = blue_labels
        self.window_size = window_size
        self.length = len(self.features) - self.window_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 输入: [idx, idx + window_size)
        # 目标: idx + window_size (下一期)
        x = self.features[idx: idx + self.window_size]  # [T, Feat_Dim]
        y_red = self.red_labels[idx + self.window_size]  # [33]
        y_blue = self.blue_labels[idx + self.window_size]  # scalar
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_red, dtype=torch.float32), torch.tensor(y_blue, dtype=torch.long)


def load_dataset(csv_path: str = "data/raw/ssq_history.csv", window_size: int = 20):
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    extractor = SSQFeatureExtractor(records)
    feat_matrix = extractor.get_feature_matrix()
    dataset = SSQDataset(feat_matrix, extractor.red_multi_hot, extractor.blue_labels, window_size=window_size)
    return dataset, extractor
