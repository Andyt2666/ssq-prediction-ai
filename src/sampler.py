#!/usr/bin/env python3
"""
src/sampler.py - 阶段 3: 概率采样生成器 (带温度参数控制)

功能:
根据 TCN 模型输出的红球 33 维概率和蓝球 16 维概率分布:
1. 引入温度系数 tau (控制概率尖锐度)
2. 无放回采样抽选 6 个红球 + 1 个蓝球
3. 高效批量生成 5,000 ~ 10,000 注候选票池
"""

import numpy as np


class SSQSampler:
    def __init__(self, red_probs: np.ndarray, blue_probs: np.ndarray, tau: float = 1.0):
        """
        red_probs: [33] 浮点概率
        blue_probs: [16] 浮点概率
        tau: 温度系数 (tau < 1.0 偏向高概率冷热极值; tau > 1.0 趋向均匀随机)
        """
        self.red_probs = np.clip(np.array(red_probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        self.blue_probs = np.clip(np.array(blue_probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        self.tau = max(0.01, float(tau))

        # 温度重归一化
        self._scaled_red = self._apply_temperature(self.red_probs, self.tau)
        self._scaled_blue = self._apply_temperature(self.blue_probs, self.tau)

    @staticmethod
    def _apply_temperature(probs: np.ndarray, tau: float) -> np.ndarray:
        logits = np.log(probs) / tau
        # Softmax 归一化为概率分布
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def sample_candidates(self, n_candidates: int = 8000, random_seed: int = None) -> list[dict]:
        """批量无放回采样候选注数池"""
        rng = np.random.default_rng(random_seed)
        red_universe = np.arange(1, 34)   # 1 ~ 33
        blue_universe = np.arange(1, 17)  # 1 ~ 16

        candidates = []
        for _ in range(n_candidates):
            # 无放回抽取 6 个红球
            sampled_reds = rng.choice(red_universe, size=6, replace=False, p=self._scaled_red)
            sampled_reds.sort()

            # 抽取 1 个蓝球
            sampled_blue = rng.choice(blue_universe, size=1, p=self._scaled_blue)[0]

            candidates.append({
                "reds": [int(x) for x in sampled_reds],
                "blue": int(sampled_blue),
                # 记录该注的联合模型对数概率得分
                "log_prob_score": float(
                    np.sum(np.log(self.red_probs[sampled_reds - 1])) +
                    np.log(self.blue_probs[sampled_blue - 1])
                )
            })

        return candidates
