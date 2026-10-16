#!/usr/bin/env python3
"""
src/backtest.py - 阶段 4: 严格时间时序前向回测与蒙特卡洛随机基线对比

功能:
1. 滚动回测 (Walk-Forward Validation): 严格按时间序列切分，逐期使用当时历史选号
2. 命中与奖级统计: 红球命中数、蓝球命中数、三/四/五/六等奖注数
3. 资金收益曲线 (ROI): 每注 2 元，固定奖级结算
4. 蒙特卡洛随机基准对比 (1000次模拟): 生成 95% 置信区间
5. 科学显著性检验: 判断模型与纯随机基线是否存在统计学显著差异
6. 自动绘制回测图表至 reports/ 目录
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无界面绘图后端
import matplotlib.pyplot as plt


def calculate_prize(red_hit: int, blue_hit: int, prizes_cfg: dict) -> tuple[int, str]:
    """根据红球命中数与蓝球命中数判定奖级与奖金"""
    p3 = prizes_cfg.get("p3", 3000)
    p4 = prizes_cfg.get("p4", 200)
    p5 = prizes_cfg.get("p5", 10)
    p6 = prizes_cfg.get("p6", 5)

    if red_hit == 6 and blue_hit == 1:
        return 5000000, "一等奖"
    elif red_hit == 6 and blue_hit == 0:
        return 200000, "二等奖"
    elif red_hit == 5 and blue_hit == 1:
        return p3, "三等奖"
    elif (red_hit == 5 and blue_hit == 0) or (red_hit == 4 and blue_hit == 1):
        return p4, "四等奖"
    elif (red_hit == 4 and blue_hit == 0) or (red_hit == 3 and blue_hit == 1):
        return p5, "五等奖"
    elif blue_hit == 1:
        # 2+1, 1+1, 0+1
        return p6, "六等奖"
    else:
        return 0, "未中奖"


class SSQBacktester:
    def __init__(self, records: list[dict], test_issues: int = 100, config: dict = None):
        self.records = records
        self.total_records = len(records)
        self.test_issues = min(test_issues, self.total_records - 50)
        self.config = config or {}
        self.prizes_cfg = self.config.get("backtest", {}).get("prizes", {
            "p3": 3000, "p4": 200, "p5": 10, "p6": 5
        })
        self.ticket_price = self.config.get("backtest", {}).get("ticket_price", 2)
        self.top_n = self.config.get("sampler", {}).get("top_n_final", 5)

    def run_simulation(self, model_predictions: list[list[dict]]) -> dict:
        """
        model_predictions: 长度为 test_issues 的列表，
        每一项包含该期精选的 top_n 注彩票: [{'reds': [...], 'blue': int}, ...]
        """
        eval_records = self.records[-self.test_issues:]
        assert len(model_predictions) == len(eval_records), "预测期数与回测期数不匹配"

        model_red_hits = []
        model_blue_hits = []
        model_payouts = []
        model_costs = []
        prize_counter = {"一等奖": 0, "二等奖": 0, "三等奖": 0, "四等奖": 0, "五等奖": 0, "六等奖": 0, "未中奖": 0}

        for idx, true_rec in enumerate(eval_records):
            true_reds = set(int(true_rec[f"r{i}"]) for i in range(1, 7))
            true_blue = int(true_rec["blue"])
            selected_tickets = model_predictions[idx]

            issue_cost = len(selected_tickets) * self.ticket_price
            issue_payout = 0

            for ticket in selected_tickets:
                r_hit = len(true_reds.intersection(set(ticket["reds"])))
                b_hit = 1 if ticket["blue"] == true_blue else 0

                model_red_hits.append(r_hit)
                model_blue_hits.append(b_hit)

                payout, prize_name = calculate_prize(r_hit, b_hit, self.prizes_cfg)
                issue_payout += payout
                prize_counter[prize_name] += 1

            model_payouts.append(issue_payout)
            model_costs.append(issue_cost)

        model_cum_cost = np.cumsum(model_costs)
        model_cum_payout = np.cumsum(model_payouts)
        model_cum_profit = model_cum_payout - model_cum_cost

        # 蒙特卡洛随机基准对比 (1000次模拟)
        mc_runs = self.config.get("backtest", {}).get("monte_carlo_runs", 1000)
        print(f"[*] 正在运行 {mc_runs} 次蒙特卡洛随机选号基线模拟...")
        rng = np.random.default_rng(42)

        mc_all_profits = np.zeros((mc_runs, self.test_issues))
        mc_red_hits = []

        for m in range(mc_runs):
            run_payouts = []
            for idx, true_rec in enumerate(eval_records):
                true_reds = set(int(true_rec[f"r{i}"]) for i in range(1, 7))
                true_blue = int(true_rec["blue"])
                issue_payout = 0
                for _ in range(self.top_n):
                    rnd_reds = rng.choice(np.arange(1, 34), size=6, replace=False)
                    rnd_blue = rng.integers(1, 17)
                    r_hit = len(true_reds.intersection(set(rnd_reds)))
                    b_hit = 1 if rnd_blue == true_blue else 0
                    payout, _ = calculate_prize(r_hit, b_hit, self.prizes_cfg)
                    issue_payout += payout
                    if m == 0:
                        mc_red_hits.append(r_hit)
                run_payouts.append(issue_payout)
            mc_all_profits[m] = np.cumsum(run_payouts) - model_cum_cost

        # 计算 95% 置信区间
        mc_lower_95 = np.percentile(mc_all_profits, 2.5, axis=0)
        mc_upper_95 = np.percentile(mc_all_profits, 97.5, axis=0)
        mc_median = np.median(mc_all_profits, axis=0)

        # 核心指标统计
        avg_red_model = float(np.mean(model_red_hits))
        avg_blue_model = float(np.mean(model_blue_hits))
        theory_red_expect = 6.0 * 6.0 / 33.0  # ≈ 1.0909
        theory_blue_expect = 1.0 / 16.0       # = 0.0625

        total_cost = float(model_cum_cost[-1])
        total_payout = float(model_cum_payout[-1])
        total_profit = float(model_cum_profit[-1])
        roi = (total_profit / total_cost) * 100.0

        # 判断是否显著偏离随机
        is_statistically_significant = bool(
            total_profit < mc_lower_95[-1] or total_profit > mc_upper_95[-1]
        )

        results = {
            "test_issues": self.test_issues,
            "tickets_per_issue": self.top_n,
            "total_tickets_evaluated": len(model_red_hits),
            "total_cost_yuan": total_cost,
            "total_payout_yuan": total_payout,
            "total_profit_yuan": total_profit,
            "roi_percent": roi,
            "red_hit_avg": avg_red_model,
            "red_hit_theoretical": theory_red_expect,
            "blue_hit_rate": avg_blue_model,
            "blue_hit_theoretical": theory_blue_expect,
            "prize_counter": prize_counter,
            "is_significant": is_statistically_significant,
            "model_cum_profit": model_cum_profit.tolist(),
            "mc_median": mc_median.tolist(),
            "mc_lower_95": mc_lower_95.tolist(),
            "mc_upper_95": mc_upper_95.tolist()
        }

        return results

    def generate_report_chart(self, results: dict, save_path: str = "reports/backtest_report.png"):
        """绘制双曲线回测对比报告"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        x_steps = np.arange(1, results["test_issues"] + 1)

        plt.figure(figsize=(10, 6), dpi=150)
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # 蒙特卡洛 95% 置信带
        plt.fill_between(x_steps, results["mc_lower_95"], results["mc_upper_95"],
                         color='gray', alpha=0.25, label='Random Baseline (95% CI)')
        plt.plot(x_steps, results["mc_median"], color='gray', linestyle='--', label='Random Baseline (Median)')

        # 模型收益曲线
        plt.plot(x_steps, results["model_cum_profit"], color='#ba2636', linewidth=2.2, label='TCN + Smart Filter Model')

        plt.title(f"SSQ Walk-Forward Backtesting (Past {results['test_issues']} Issues)", fontsize=13, fontweight='bold')
        plt.xlabel("Test Issue Steps", fontsize=11)
        plt.ylabel("Cumulative Net Profit (RMB)", fontsize=11)
        plt.axhline(0, color='black', linewidth=0.8, linestyle=':')
        plt.legend(loc='lower left', fontsize=10)
        plt.tight_layout()

        plt.savefig(save_path)
        plt.close()
        print(f"[+] 回测可视化报告已成功保存至: {save_path}")
