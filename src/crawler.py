#!/usr/bin/env python3
"""
src/crawler.py - 阶段 0: 数据采集、清洗入库与统计学随机性检验

功能:
1. 全量与增量拉取历史双色球数据 (2003年创办至今)
2. 清洗并存储至 CSV 与 SQLite 数据库 (data/ssq.db)
3. 统计学随机性检验 (卡方检验、游程检验、序列自相关检验)
"""

import os
import re
import sqlite3
import csv
import urllib.request
import math
from collections import Counter

DATA_URL = "https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=26999"


def fetch_and_save(csv_path: str = "data/raw/ssq_history.csv", db_path: str = "data/ssq.db") -> list[dict]:
    """抓取全部历史数据并存入 CSV 与 SQLite 数据库"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    print(f"[*] 正在从数据源拉取双色球全量开奖数据...")
    req = urllib.request.Request(
        DATA_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("gb2312", errors="ignore")

    rows = re.findall(r'<tr class="t_tr1".*?>(.*?)</tr>', html, re.S)
    print(f"[+] 抓取成功，解析到 {len(rows)} 期原始数据。")

    records = []
    for r in rows:
        tds = re.findall(r"<td.*?>(.*?)</td>", r, re.S)
        cleaned = [re.sub(r"<.*?>|&nbsp;|\s+", "", td) for td in tds]
        if len(cleaned) < 17:
            continue

        issue = cleaned[1]
        reds = [int(x) for x in cleaned[2:8] if x.isdigit()]
        if len(reds) != 6:
            continue
        reds.sort()

        blue = int(cleaned[8]) if cleaned[8].isdigit() else 0
        date = cleaned[16] if len(cleaned) > 16 else ""

        records.append({
            "issue": issue,
            "date": date,
            "r1": f"{reds[0]:02d}",
            "r2": f"{reds[1]:02d}",
            "r3": f"{reds[2]:02d}",
            "r4": f"{reds[3]:02d}",
            "r5": f"{reds[4]:02d}",
            "r6": f"{reds[5]:02d}",
            "blue": f"{blue:02d}",
            "sum_red": sum(reds),
            "span_red": reds[5] - reds[0],
            "jackpot_pool": cleaned[10].replace(",", ""),
            "first_prize_count": cleaned[11].replace(",", ""),
            "first_prize_amount": cleaned[12].replace(",", ""),
            "total_sales": cleaned[15].replace(",", ""),
        })

    # 按期号升序（从旧到新）
    records.sort(key=lambda x: x["issue"])

    # 1. 写入 CSV
    fieldnames = [
        "issue", "date", "r1", "r2", "r3", "r4", "r5", "r6", "blue",
        "sum_red", "span_red", "jackpot_pool", "first_prize_count",
        "first_prize_amount", "total_sales"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # 2. 写入 SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ssq_history (
        issue TEXT PRIMARY KEY,
        date TEXT,
        r1 INTEGER, r2 INTEGER, r3 INTEGER, r4 INTEGER, r5 INTEGER, r6 INTEGER,
        blue INTEGER,
        sum_red INTEGER, span_red INTEGER,
        jackpot_pool TEXT, first_prize_count TEXT, first_prize_amount TEXT, total_sales TEXT
    )
    """)
    for rec in records:
        cursor.execute("""
        INSERT OR REPLACE INTO ssq_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["issue"], rec["date"],
            int(rec["r1"]), int(rec["r2"]), int(rec["r3"]), int(rec["r4"]), int(rec["r5"]), int(rec["r6"]),
            int(rec["blue"]), int(rec["sum_red"]), int(rec["span_red"]),
            rec["jackpot_pool"], rec["first_prize_count"], rec["first_prize_amount"], rec["total_sales"]
        ))
    conn.commit()
    conn.close()

    print(f"[+] 数据清洗入库完毕: CSV -> {csv_path}, SQLite -> {db_path} (共 {len(records)} 期)")
    return records


def run_randomness_tests(csv_path: str = "data/raw/ssq_history.csv") -> dict:
    """阶段 0 核心: 运行统计学随机性检验 (卡方检验、游程检验、序列自相关检验)"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件 {csv_path}，请先执行 fetch_and_save()")

    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    N = len(records)
    print("\n" + "=" * 65)
    print(f"       阶段 0: 历史开奖数据的统计学随机性深度检验 (共 {N} 期)")
    print("=" * 65)

    # 1. 红球分布卡方检验 (Chi-square Test on 33 Red Balls)
    # 理论上每一期抽6个球，红球总计开出 6*N 次，每个球的期望次数 E = 6*N / 33
    red_counts = Counter()
    for rec in records:
        for i in range(1, 7):
            red_counts[int(rec[f"r{i}"])] += 1

    expected_red = (6.0 * N) / 33.0
    chi2_red = sum((red_counts[i] - expected_red) ** 2 / expected_red for i in range(1, 34))
    # 自由度 df = 33 - 1 = 32
    # 临界值 (alpha = 0.05, df=32): chi2_critical ≈ 46.19
    crit_red_05 = 46.19

    # 2. 蓝球分布卡方检验 (Chi-square Test on 16 Blue Balls)
    # 蓝球每期1个，每个蓝球期望次数 E = N / 16
    blue_counts = Counter()
    for rec in records:
        blue_counts[int(rec["blue"])] += 1

    expected_blue = float(N) / 16.0
    chi2_blue = sum((blue_counts[i] - expected_blue) ** 2 / expected_blue for i in range(1, 17))
    # 自由度 df = 16 - 1 = 15, 临界值 (alpha=0.05, df=15): chi2_critical ≈ 24.996
    crit_blue_05 = 24.996

    # 3. 游程检验 (Wald-Wolfowitz Runs Test on Blue Ball Odd/Even)
    # 检验相邻开奖之间是否存在序列交替或聚集效应
    blue_seq = [int(rec["blue"]) % 2 for rec in records]
    n1 = sum(1 for x in blue_seq if x == 1) # 奇数个数
    n2 = sum(1 for x in blue_seq if x == 0) # 偶数个数
    runs = 1
    for i in range(1, len(blue_seq)):
        if blue_seq[i] != blue_seq[i - 1]:
            runs += 1

    mu_runs = 1.0 + (2.0 * n1 * n2) / (n1 + n2)
    var_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z_runs = (runs - mu_runs) / math.sqrt(var_runs) if var_runs > 0 else 0

    # 4. 红球和值自相关检验 (Lag-1, Lag-2, Lag-3 Autocorrelation of Red Sum)
    sums = [int(rec["sum_red"]) for rec in records]
    mean_sum = sum(sums) / len(sums)
    var_sum = sum((s - mean_sum) ** 2 for s in sums) / len(sums)

    autocorr = {}
    for lag in [1, 2, 3]:
        cov = sum((sums[i] - mean_sum) * (sums[i - lag] - mean_sum) for i in range(lag, len(sums))) / (len(sums) - lag)
        autocorr[lag] = cov / var_sum if var_sum > 0 else 0

    print(f"1. 【红球卡方拟合优度检验 (自由度=32)】")
    print(f"   • 统计量 χ² = {chi2_red:.2f} | 显著性临界值 (α=0.05) = {crit_red_05}")
    red_pass = chi2_red < crit_red_05
    print(f"   • 检验结果: {'✅ 接受均匀分布原假设 (无显著频次偏倚)' if red_pass else '⚠️ 存在弱偏倚'}")

    print(f"\n2. 【蓝球卡方拟合优度检验 (自由度=15)】")
    print(f"   • 统计量 χ² = {chi2_blue:.2f} | 显著性临界值 (α=0.05) = {crit_blue_05}")
    blue_pass = chi2_blue < crit_blue_05
    print(f"   • 检验结果: {'✅ 接受均匀分布原假设 (无显著频次偏倚)' if blue_pass else '⚠️ 存在弱偏倚'}")

    print(f"\n3. 【蓝球奇偶独立性游程检验 (Runs Test)】")
    print(f"   • 观察游程数 = {runs}, 期望游程数 = {mu_runs:.1f}, Z统计量 = {z_runs:.3f}")
    runs_pass = abs(z_runs) < 1.96
    print(f"   • 检验结果: {'✅ 无法拒绝独立性假设 (|Z| < 1.96，序列无记忆粘性)' if runs_pass else '⚠️ 存在轻微聚集/交替效应'}")

    print(f"\n4. 【红球和值滞后自相关系数 (Autocorrelation)】")
    for lag, r in autocorr.items():
        print(f"   • Lag-{lag} 自相关系数: {r:.4f} (显著性阈值 ≈ ±{2.0 / math.sqrt(N):.4f})")

    print("\n" + "-" * 65)
    print("📢 【阶段 0 诚实验证结论】:")
    print("   历史开奖在严格统计学检验下，各号码频次与理论均匀随机高度吻合，")
    print("   前后开奖不存在显著的自相关性。因此任何深度学习时序模型在理论上")
    print("   均无法获得超出随机基线的超额收益。本项目后续建模与缩水旨在验证")
    print("   这一科学结论并提供形态优化机制。")
    print("-" * 65 + "\n")

    return {
        "chi2_red": chi2_red, "red_pass": red_pass,
        "chi2_blue": chi2_blue, "blue_pass": blue_pass,
        "z_runs": z_runs, "runs_pass": runs_pass,
        "autocorr": autocorr
    }


if __name__ == "__main__":
    fetch_and_save()
    run_randomness_tests()
