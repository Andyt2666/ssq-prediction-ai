#!/usr/bin/env python3
"""
SSQ History Data Fetcher
Fetches all historical Double Color Ball (双色球) lottery data from 2003 to current.
No external dependencies required (uses standard library urllib and csv).
"""

import csv
import os
import re
import sys
import urllib.request

DATA_URL = "https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=26999"


def fetch_ssq_history(save_path: str = None) -> list[dict]:
    """Fetch all historical SSQ data and return as list of dicts.

    Optionally saves to CSV.
    """
    if save_path is None:
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..")
        )
        save_path = os.path.join(project_root, "data/raw/ssq_history.csv")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[*] 正在从 500.com 拉取历史双色球全量开奖数据...")
    req = urllib.request.Request(
        DATA_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("gb2312", errors="ignore")

    rows = re.findall(r'<tr class="t_tr1".*?>(.*?)</tr>', html, re.S)
    print(f"[+] 抓取成功，解析到 {len(rows)} 期历史开奖记录。")

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

    # Sort in chronological order (oldest to newest)
    records.sort(key=lambda x: x["issue"])

    # Write to CSV
    fieldnames = [
        "issue",
        "date",
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
        "r6",
        "blue",
        "sum_red",
        "span_red",
        "jackpot_pool",
        "first_prize_count",
        "first_prize_amount",
        "total_sales",
    ]

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(
        f"[+] 成功保存 {len(records)} 期开奖数据至: {os.path.abspath(save_path)}"
    )
    if records:
        latest = records[-1]
        print(
            f"[*] 最新期号: {latest['issue']} (开奖日: {latest['date']}) -> 红球:"
            f" {latest['r1']},{latest['r2']},{latest['r3']},{latest['r4']},{latest['r5']},{latest['r6']}"
            f" | 蓝球: {latest['blue']}"
        )

    return records


if __name__ == "__main__":
    fetch_ssq_history()
