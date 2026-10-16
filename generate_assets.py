#!/usr/bin/env python3
"""
generate_assets.py - 生成全套高分辨率科研级可视化图表
用于 GitHub README 与项目展示
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

os.makedirs("assets", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 1. 载入 3515 期历史数据
with open("data/raw/ssq_history.csv", "r", encoding="utf-8") as f:
    records = list(csv.DictReader(f))

N = len(records)
print(f"[*] 载入 {N} 期历史数据，开始生成高清可视化资产...")

# -------------------------------------------------------------
# 图表 1: 历史全景统计学分析看板 (eda_dashboard.png)
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200)
fig.suptitle(f"China Welfare Lottery SSQ - 3,515 Issues Full Historical EDA Dashboard (2003-2026)", fontsize=16, fontweight='bold', y=0.98)

# (1) 红球出号频次
red_counts = np.zeros(33)
for r in records:
    for k in range(1, 7):
        red_counts[int(r[f"r{k}"]) - 1] += 1

red_exp = (6.0 * N) / 33.0
bars = axes[0, 0].bar(range(1, 34), red_counts, color="#e63946", alpha=0.85, edgecolor="#1d3557", linewidth=0.8)
axes[0, 0].axhline(red_exp, color="#1d3557", linestyle="--", linewidth=1.5, label=f"Theoretical Expectation ({red_exp:.1f})")
axes[0, 0].set_title("Red Ball (01-33) Frequency Distribution", fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel("Red Ball Number", fontsize=11)
axes[0, 0].set_ylabel("Occurrence Count", fontsize=11)
axes[0, 0].set_xticks(range(1, 34, 2))
axes[0, 0].legend(loc="upper right", fontsize=10)
axes[0, 0].set_ylim(500, 750)

# (2) 蓝球出号频次
blue_counts = np.zeros(16)
for r in records:
    blue_counts[int(r["blue"]) - 1] += 1

blue_exp = float(N) / 16.0
axes[0, 1].bar(range(1, 17), blue_counts, color="#1d3557", alpha=0.85, edgecolor="#457b9d", linewidth=0.8)
axes[0, 1].axhline(blue_exp, color="#e63946", linestyle="--", linewidth=1.5, label=f"Theoretical Expectation ({blue_exp:.1f})")
axes[0, 1].set_title("Blue Ball (01-16) Frequency Distribution", fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel("Blue Ball Number", fontsize=11)
axes[0, 1].set_ylabel("Occurrence Count", fontsize=11)
axes[0, 1].set_xticks(range(1, 17))
axes[0, 1].legend(loc="upper right", fontsize=10)
axes[0, 1].set_ylim(180, 260)

# (3) 红球和值分布与正态拟合
sums = np.array([int(r["sum_red"]) for r in records])
mu_sum, std_sum = norm.fit(sums)
n_bins, bins, patches = axes[1, 0].hist(sums, bins=35, density=True, alpha=0.75, color="#2a9d8f", edgecolor="#264653")
xmin, xmax = axes[1, 0].get_xlim()
x_axis = np.linspace(xmin, xmax, 200)
p = norm.pdf(x_axis, mu_sum, std_sum)
axes[1, 0].plot(x_axis, p, '#e76f51', linewidth=2.5, label=f"Normal Fit (μ={mu_sum:.1f}, σ={std_sum:.1f})")
axes[1, 0].axvline(101.0, color="#1d3557", linestyle=":", linewidth=1.8, label="Mean Center (101.0)")
axes[1, 0].set_title("Red Balls Sum Value Gaussian Distribution", fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel("Red Sum (6 Balls)", fontsize=11)
axes[1, 0].set_ylabel("Probability Density", fontsize=11)
axes[1, 0].legend(loc="upper right", fontsize=10)

# (4) 奇偶比历史分布 (环形饼图)
odd_ratios = {}
for r in records:
    odds = sum(1 for k in range(1, 7) if int(r[f"r{k}"]) % 2 == 1)
    key = f"{odds}:{6-odds}"
    odd_ratios[key] = odd_ratios.get(key, 0) + 1

labels = ["3:3 (Golden)", "4:2", "2:4", "5:1", "1:5", "6:0 / 0:6"]
counts = [
    odd_ratios.get("3:3", 0),
    odd_ratios.get("4:2", 0),
    odd_ratios.get("2:4", 0),
    odd_ratios.get("5:1", 0),
    odd_ratios.get("1:5", 0),
    odd_ratios.get("6:0", 0) + odd_ratios.get("0:6", 0)
]
colors = ['#e63946', '#f4a261', '#e76f51', '#2a9d8f', '#457b9d', '#6c757d']
wedges, texts, autotexts = axes[1, 1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=140,
                                          colors=colors, pctdistance=0.8,
                                          wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
axes[1, 1].set_title("Red Balls Odd/Even Ratio Distribution", fontsize=13, fontweight='bold')
for autotext in autotexts:
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("assets/eda_dashboard.png", dpi=200)
plt.close()
print("[+] 生成: assets/eda_dashboard.png")

# -------------------------------------------------------------
# 图表 2: 四重异质时空图拓扑邻接热力图 (graph_topologies.png)
# -------------------------------------------------------------
from src.model_stgnn import build_domain_adj_matrices, SSQComplexSTGNN
import torch

A_domain, A_co, A_trans = build_domain_adj_matrices(records)
# 提取已训练模型的自适应图
model = SSQComplexSTGNN(adj_domain=A_domain, adj_co=A_co, adj_trans=A_trans)
if os.path.exists("data/stgnn_model.pth"):
    model.load_state_dict(torch.load("data/stgnn_model.pth", map_location="cpu"))
A_adaptive = model._get_adaptive_adj().detach().cpu().numpy()

fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=200)
fig.suptitle("ST-GNN-Vision: 4-Fold Heterogeneous Topological Graph Adjacency Matrices (33x33)", fontsize=15, fontweight='bold', y=1.02)

adjs = [
    (A_domain, "A_domain: Number-Theoretic\n(Consecutive, Last-Digit, Zone)", "YlGnBu"),
    (A_co, "A_co: Empirical Co-occurrence\n(3,515 Issues Joint Appearances)", "YlOrRd"),
    (A_trans, "A_trans: Markov Transition\n(Issue t-1 -> Issue t Probabilities)", "Purples"),
    (A_adaptive, "A_adaptive: Learnable Latent Graph\n(End-to-End Latent Embeddings)", "viridis")
]

for idx, (adj, title, cmap) in enumerate(adjs):
    im = axes[idx].imshow(adj, cmap=cmap, aspect='equal', interpolation='nearest')
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel("Red Ball Node (01-33)", fontsize=10)
    if idx == 0:
        axes[idx].set_ylabel("Red Ball Node (01-33)", fontsize=10)
    fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("assets/graph_topologies.png", dpi=200)
plt.close()
print("[+] 生成: assets/graph_topologies.png")

# -------------------------------------------------------------
# 图表 3: 走势图视觉点阵切片示例 (trend_matrix_vision.png)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
# 提取最近 40 期的开奖走势矩阵
recent_40 = np.zeros((40, 33))
for i, r in enumerate(records[-40:]):
    reds = [int(r[f"r{k}"]) - 1 for k in range(1, 7)]
    recent_40[i, reds] = 1.0

ax.imshow(recent_40, cmap="Reds", aspect='auto', interpolation='nearest')
ax.set_title("Visual Trend Chart 2D Grid Stream (Recent 40 Issues x 33 Red Balls)", fontsize=13, fontweight='bold')
ax.set_xlabel("Red Ball Node (01-33)", fontsize=11)
ax.set_ylabel("Historical Issues (Past 40 -> Latest)", fontsize=11)
ax.set_xticks(range(33))
ax.set_xticklabels([f"{x+1:02d}" for x in range(33)], fontsize=8)
# 网格线
ax.set_xticks(np.arange(-.5, 33, 1), minor=True)
ax.set_yticks(np.arange(-.5, 40, 1), minor=True)
ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig("assets/trend_matrix_vision.png", dpi=200)
plt.close()
print("[+] 生成: assets/trend_matrix_vision.png")

# -------------------------------------------------------------
# 图表 4: 综合科研回溯对比全景图 (backtest_benchmark.png)
# -------------------------------------------------------------
import json
with open("reports/backtest_summary.json", "r") as f:
    tcn_res = json.load(f)
with open("reports/backtest_stgnn_summary.json", "r") as f:
    stgnn_res = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=200)
fig.suptitle("Historical Walk-Forward Backtesting Benchmark: ST-GNN vs TCN vs 1,000 Monte Carlo Runs", fontsize=15, fontweight='bold', y=1.02)

# (1) 累计资金损益曲线 vs 95% 置信带
x_steps = np.arange(1, 101)
axes[0].fill_between(x_steps, stgnn_res["mc_lower_95"], stgnn_res["mc_upper_95"],
                     color='#6c757d', alpha=0.25, label='Random Baseline (95% CI Band)')
axes[0].plot(x_steps, stgnn_res["mc_median"], color='#495057', linestyle='--', linewidth=1.5, label='Random Baseline (Median)')
axes[0].plot(x_steps, tcn_res["model_cum_profit"], color='#f4a261', linewidth=2.0, label=f"TCN Baseline (ROI: {tcn_res['roi_percent']:.1f}%)")
axes[0].plot(x_steps, stgnn_res["model_cum_profit"], color='#e63946', linewidth=2.4, label=f"ST-GNN-Vision (ROI: {stgnn_res['roi_percent']:.1f}%)")
axes[0].axhline(0, color='black', linewidth=0.8, linestyle=':')
axes[0].set_title("Cumulative Net Profit (RMB) Walk-Forward Curve", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Evaluated Issues (Past 100 Issues)", fontsize=11)
axes[0].set_ylabel("Net Profit / Loss (RMB)", fontsize=11)
axes[0].legend(loc="lower left", fontsize=9)

# (2) 红蓝球命中率对比柱状图
labels = ['Red Ball Avg Hit Count', 'Blue Ball Hit Rate (%)']
model_stgnn_hits = [stgnn_res['red_hit_avg'], stgnn_res['blue_hit_rate'] * 100]
model_tcn_hits = [tcn_res['red_hit_avg'], tcn_res['blue_hit_rate'] * 100]
theory_hits = [6.0 * 6.0 / 33.0, (1.0 / 16.0) * 100]

x = np.arange(len(labels))
width = 0.25
axes[1].bar(x - width, theory_hits, width, label='Theoretical Expectation', color='#457b9d', alpha=0.9)
axes[1].bar(x, model_tcn_hits, width, label='TCN Baseline', color='#f4a261', alpha=0.9)
axes[1].bar(x + width, model_stgnn_hits, width, label='ST-GNN-Vision', color='#e63946', alpha=0.9)
axes[1].set_title("Average Hit Count & Rate Benchmark", fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=11, fontweight='bold')
axes[1].legend(loc="upper right", fontsize=9)

# (3) 奖级命中数量对比 (五等奖与六等奖)
prize_labels = ['5th Prize (10 RMB)', '6th Prize (5 RMB)', 'Total Payout (RMB)']
p_tcn = [tcn_res['prize_counter']['五等奖'], tcn_res['prize_counter']['六等奖'], tcn_res['total_payout_yuan']]
p_stgnn = [stgnn_res['prize_counter']['五等奖'], stgnn_res['prize_counter']['六等奖'], stgnn_res['total_payout_yuan']]

x_p = np.arange(len(prize_labels))
width_p = 0.35
axes[2].bar(x_p - width_p/2, p_tcn, width_p, label='TCN Baseline', color='#f4a261', alpha=0.9)
axes[2].bar(x_p + width_p/2, p_stgnn, width_p, label='ST-GNN-Vision', color='#e63946', alpha=0.9)
axes[2].set_title("Prizes & Total Returns (100 Issues / 500 Tickets)", fontsize=12, fontweight='bold')
axes[2].set_xticks(x_p)
axes[2].set_xticklabels(prize_labels, fontsize=10, fontweight='bold')
axes[2].legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("assets/backtest_benchmark.png", dpi=200)
plt.close()
print("[+] 生成: assets/backtest_benchmark.png")
