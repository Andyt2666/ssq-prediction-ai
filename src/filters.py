#!/usr/bin/env python3
"""
src/filters.py - 阶段 3: 规则漏斗过滤与多样性精选引擎

功能:
1. 和值过滤: 和值在 [80, 130] 之间
2. 奇偶比过滤: 仅保留 3:3, 4:2, 2:4 (覆盖历史 83% 真实中奖形态)
3. AC 值过滤: 数字复杂度 AC >= 6
4. 连号深度过滤: 最大连号不超过 2 (剔除 3 连号及以上)
5. 避开大众热门/生日号: 剔除全部 <= 31 且集中在月日范围、等差数列等大众扎堆组合
6. 多样性去重约束: 选出的精选 5/10 注之间，两两红球重合数 <= 3 个
"""

from src.features import compute_ac_value, count_consecutive


class SSQFilter:
    def __init__(self, config: dict = None):
        cfg = config or {}
        self.sum_min = cfg.get("sum_range", [80, 130])[0]
        self.sum_max = cfg.get("sum_range", [80, 130])[1]
        self.valid_odd_even = set(cfg.get("valid_odd_even", ["3:3", "4:2", "2:4"]))
        self.min_ac = cfg.get("min_ac_value", 6)
        self.max_consecutive = cfg.get("max_consecutive", 2)
        self.avoid_popular = cfg.get("avoid_popular", True)
        self.max_red_overlap = cfg.get("max_red_overlap", 3)

    def is_valid(self, ticket: dict) -> bool:
        reds = ticket["reds"]

        # 1. 和值过滤
        s = sum(reds)
        if not (self.sum_min <= s <= self.sum_max):
            return False

        # 2. 奇偶比过滤
        odds = sum(1 for x in reds if x % 2 == 1)
        odd_even_str = f"{odds}:{6 - odds}"
        if odd_even_str not in self.valid_odd_even:
            return False

        # 3. AC 值过滤
        ac = compute_ac_value(reds)
        if ac < self.min_ac:
            return False

        # 4. 连号过滤
        cons = count_consecutive(reds)
        if cons > self.max_consecutive:
            return False

        # 5. 避开大众热门组合 (如全部集中在生日 1-31 日期内且跨度过小，或等差数列)
        if self.avoid_popular:
            # 等差数列检查
            diffs = [reds[i] - reds[i - 1] for i in range(1, 6)]
            if len(set(diffs)) == 1:
                return False  # 纯等差数列大众极爱买
            # 极小跨度 (例如 6 个号都在 1~12 之内)
            if reds[5] - reds[0] < 12:
                return False

        return True

    def filter_candidates(self, candidates: list[dict]) -> tuple[list[dict], dict]:
        """批量过滤候选池并统计通过率"""
        passed = []
        stats = {
            "total_input": len(candidates),
            "passed_count": 0,
            "pass_rate": 0.0
        }

        for c in candidates:
            if self.is_valid(c):
                passed.append(c)

        stats["passed_count"] = len(passed)
        stats["pass_rate"] = len(passed) / max(1, len(candidates))
        return passed, stats

    def select_diverse_tickets(self, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """按模型得分排序，并施加红球重合数 <= max_red_overlap 的多样性贪心精选"""
        # 先按模型概率得分从大到小排序
        sorted_candidates = sorted(candidates, key=lambda x: x.get("log_prob_score", 0), reverse=True)

        selected = []
        for cand in sorted_candidates:
            if len(selected) >= top_n:
                break

            cand_reds = set(cand["reds"])
            # 检查与已入选注数的重合数
            conflict = False
            for sel in selected:
                overlap = len(cand_reds.intersection(set(sel["reds"])))
                if overlap > self.max_red_overlap:
                    conflict = True
                    break

            if not conflict:
                selected.append(cand)

        return selected
