#!/usr/bin/env python3
"""
SSQ Basic Statistical Analysis
Computes hot/cold numbers, current omission gaps, sum distribution, and odd/even ratios.
Uses standard Python library.
"""

import csv
import os
from collections import Counter


def analyze_stats(csv_path: str = None):
    if csv_path is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..")
        )
        csv_path = os.path.join(project_root, "data/raw/ssq_history.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Run fetcher.py first.")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = list(reader)

    total_issues = len(records)
    print("=" * 60)
    print(f"  双色球历史全量开奖统计分析 (共 {total_issues} 期)")
    print("=" * 60)

    # 1. Red Ball & Blue Ball Frequencies
    red_counter = Counter()
    blue_counter = Counter()

    # Track last seen index for omission (0 = last issue)
    red_last_seen = {i: -1 for i in range(1, 34)}
    blue_last_seen = {i: -1 for i in range(1, 17)}

    sum_list = []
    odd_counts = Counter()

    for idx, rec in enumerate(records):
        reds = [int(rec[f"r{i}"]) for i in range(1, 7)]
        blue = int(rec["blue"])
        sum_list.append(int(rec["sum_red"]))

        # Count odds
        odds = sum(1 for r in reds if r % 2 == 1)
        odd_counts[f"{odds}:{6-odds}"] += 1

        for r in reds:
            red_counter[r] += 1
            red_last_seen[r] = idx

        blue_counter[blue] += 1
        blue_last_seen[blue] = idx

    # Calculate current omission (gap from latest issue)
    latest_idx = total_issues - 1
    red_omission = {r: latest_idx - red_last_seen[r] for r in range(1, 34)}
    blue_omission = {b: latest_idx - blue_last_seen[b] for b in range(1, 17)}

    # Hot & Cold Reds
    sorted_reds = red_counter.most_common()
    print("\n[🔴 红球历史出号频次排行]")
    print(
        "最热 Top 5 红球:",
        ", ".join(
            f"{num:02d}号({cnt}次)" for num, cnt in sorted_reds[:5]
        ),
    )
    print(
        "最冷 Top 5 红球:",
        ", ".join(
            f"{num:02d}号({cnt}次)" for num, cnt in sorted_reds[-5:]
        ),
    )

    # Hot & Cold Blues
    sorted_blues = blue_counter.most_common()
    print("\n[🔵 蓝球历史出号频次排行]")
    print(
        "最热 Top 3 蓝球:",
        ", ".join(
            f"{num:02d}号({cnt}次)" for num, cnt in sorted_blues[:3]
        ),
    )
    print(
        "最冷 Top 3 蓝球:",
        ", ".join(
            f"{num:02d}号({cnt}次)" for num, cnt in sorted_blues[-3:]
        ),
    )

    # Current biggest omissions (最长未开出)
    sorted_red_omissions = sorted(
        red_omission.items(), key=lambda x: x[1], reverse=True
    )
    sorted_blue_omissions = sorted(
        blue_omission.items(), key=lambda x: x[1], reverse=True
    )
    print("\n[⏳ 当前遗漏最大号码 (连续未开出期数)]")
    print(
        "红球当前遗漏王 Top 5:",
        ", ".join(
            f"{num:02d}号(已遗漏{gap}期)"
            for num, gap in sorted_red_omissions[:5]
        ),
    )
    print(
        "蓝球当前遗漏王 Top 3:",
        ", ".join(
            f"{num:02d}号(已遗漏{gap}期)"
            for num, gap in sorted_blue_omissions[:3]
        ),
    )

    # Odd/Even ratio
    print("\n[⚖️ 红球奇偶比历史分布]")
    for ratio, cnt in odd_counts.most_common():
        pct = (cnt / total_issues) * 100
        print(f"  奇偶比 {ratio}: {cnt} 期 ({pct:.1f}%)")

    # Sum Stats
    avg_sum = sum(sum_list) / len(sum_list)
    print(f"\n[📊 和值统计] 平均和值: {avg_sum:.1f}, 最小: {min(sum_list)}, 最大: {max(sum_list)}")
    print("=" * 60)


if __name__ == "__main__":
    analyze_stats()
