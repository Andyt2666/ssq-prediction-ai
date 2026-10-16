#!/usr/bin/env python3
"""
main.py - 双色球 AI 选号与实验系统主控制入口 (CLI)

支持命令:
  python main.py fetch                # 阶段 0: 抓取最新数据、清洗入库并执行随机性检验
  python main.py train [--epochs 25]  # 阶段 2: 训练 TCN 因果卷积双头模型
  python main.py predict [--num 5]    # 阶段 3: 执行 AI 采样 + 规则漏斗过滤，输出今日精选号码
  python main.py backtest [--issues 100] # 阶段 4: 滚动回测对比 1000 次蒙特卡洛随机基线
"""

import argparse
import json
import os
import sys
import yaml
import torch
import numpy as np

from src.crawler import fetch_and_save, run_randomness_tests
from src.features import load_dataset, SSQFeatureExtractor
from src.model_tcn import SSQTCNModel, compute_loss
from src.sampler import SSQSampler
from src.filters import SSQFilter
from src.backtest import SSQBacktester

DISCLAIMER = """
===============================================================================
⚠️  【理性购彩与科学免责声明】:
1. 双色球在物理和数学本质上是独立同分布（i.i.d.）的随机事件，单注头奖概率为 1/17,721,088。
2. 历史开奖与未来开奖无统计学记忆性，任何深度学习模型均无法突破概率极限提高真实中奖胜率。
3. 本项目输出的号码形态优化（和值/奇偶/连号缩水）仅用于工程实践与科学实验探讨，切勿盲目投注！
===============================================================================
"""


def get_device():
    if torch.cuda.is_available():
        try:
            d = torch.nn.Conv1d(2, 2, 3).cuda()
            _ = d(torch.randn(1, 2, 5).cuda())
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def load_config(config_path="configs/default.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def cmd_fetch(args, cfg):
    csv_path = cfg.get("data", {}).get("csv_path", "data/raw/ssq_history.csv")
    db_path = cfg.get("data", {}).get("db_path", "data/ssq.db")
    fetch_and_save(csv_path, db_path)
    run_randomness_tests(csv_path)


def cmd_train(args, cfg):
    csv_path = cfg.get("data", {}).get("csv_path", "data/raw/ssq_history.csv")
    if not os.path.exists(csv_path):
        print(f"[*] 未检测到数据，先执行数据拉取...")
        fetch_and_save(csv_path)

    w_size = cfg.get("features", {}).get("window_size", 20)
    dataset, extractor = load_dataset(csv_path, window_size=w_size)

    # 划分训练集和测试集 (保留最后 100 期作为独立测试集，杜绝未来泄漏)
    test_size = 100
    train_size = len(dataset) - test_size
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(dataset)))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    batch_size = cfg.get("model", {}).get("batch_size", 32)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    device = get_device()
    print(f"[*] 训练计算设备: {device}")

    in_dim = dataset.features.shape[1]
    model = SSQTCNModel(
        input_dim=in_dim,
        num_channels=cfg.get("model", {}).get("num_channels", [64, 64, 64]),
        kernel_size=cfg.get("model", {}).get("kernel_size", 3),
        dropout=cfg.get("model", {}).get("dropout", 0.2)
    ).to(device)

    epochs = args.epochs or cfg.get("model", {}).get("epochs", 25)
    lr = cfg.get("model", {}).get("lr", 0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[*] 开始训练 TCN 双头模型 (共 {epochs} 轮)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for x, y_red, y_blue in train_loader:
            x, y_red, y_blue = x.to(device), y_red.to(device), y_blue.to(device)
            optimizer.zero_grad()
            r_logits, b_logits = model(x)
            loss, r_l, b_l = compute_loss(r_logits, b_logits, y_red, y_blue)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(x)

        avg_train = total_train_loss / train_size

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for x, y_red, y_blue in test_loader:
                    x, y_red, y_blue = x.to(device), y_red.to(device), y_blue.to(device)
                    r_logits, b_logits = model(x)
                    loss, _, _ = compute_loss(r_logits, b_logits, y_red, y_blue)
                    total_test_loss += loss.item() * len(x)
            avg_test = total_test_loss / test_size
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - 训练损失: {avg_train:.4f} | 验证损失: {avg_test:.4f}")

    # 保存模型权重
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "tcn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[+] 模型权重已保存至: {model_path}")


def cmd_predict(args, cfg):
    csv_path = cfg.get("data", {}).get("csv_path", "data/raw/ssq_history.csv")
    w_size = cfg.get("features", {}).get("window_size", 20)
    dataset, extractor = load_dataset(csv_path, window_size=w_size)

    # 提取最近 w_size 期的特征输入
    recent_x = extractor.get_feature_matrix()[-w_size:]
    in_dim = recent_x.shape[1]
    device = get_device()

    model = SSQTCNModel(input_dim=in_dim).to(device)
    model_path = "data/tcn_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[*] 已载入预训练 TCN 权重: {model_path}")
    else:
        print("[!] 未找到训练好的权重，使用冷热频次加权先验进行采样...")

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(recent_x, dtype=torch.float32).unsqueeze(0).to(device)
        r_logits, b_logits = model(x_tensor)
        red_probs = torch.sigmoid(r_logits).squeeze(0).cpu().numpy()
        blue_probs = torch.softmax(b_logits, dim=-1).squeeze(0).cpu().numpy()

    # 采样候选池
    tau = cfg.get("sampler", {}).get("tau", 1.0)
    n_cands = cfg.get("sampler", {}).get("num_candidates", 8000)
    sampler = SSQSampler(red_probs, blue_probs, tau=tau)
    candidates = sampler.sample_candidates(n_candidates=n_cands)

    # 规则漏斗过滤
    f_engine = SSQFilter(cfg.get("filters", {}))
    passed, stats = f_engine.filter_candidates(candidates)
    print(f"[*] 漏斗过滤完成: 原始采样 {stats['total_input']} 注 -> 符合规则 {stats['passed_count']} 注 (合格率 {stats['pass_rate']*100:.1f}%)")

    top_n = args.num or cfg.get("sampler", {}).get("top_n_final", 5)
    selected = f_engine.select_diverse_tickets(passed, top_n=top_n)

    print("\n" + "=" * 65)
    print(f"       🎯 【AI 概率采样 + 智能缩水精选 {len(selected)} 注推荐】")
    print("=" * 65)
    for idx, t in enumerate(selected, 1):
        red_str = " ".join(f"{x:02d}" for x in t["reds"])
        blue_str = f"{t['blue']:02d}"
        s_red = sum(t["reds"])
        odds = sum(1 for x in t["reds"] if x % 2 == 1)
        print(f"  第 {idx} 注:  🔴 红球 [ {red_str} ]  🔵 蓝球 [ {blue_str} ]  | 和值: {s_red:03d} 奇偶: {odds}:{6-odds}")
    print("=" * 65)
    print(DISCLAIMER)


def cmd_backtest(args, cfg):
    csv_path = cfg.get("data", {}).get("csv_path", "data/raw/ssq_history.csv")
    w_size = cfg.get("features", {}).get("window_size", 20)
    dataset, extractor = load_dataset(csv_path, window_size=w_size)

    test_issues = args.issues or cfg.get("backtest", {}).get("test_issues", 100)
    print(f"[*] 正在启动对过去 {test_issues} 期的 Walk-Forward 滚动前向回测...")

    all_feat_matrix = extractor.get_feature_matrix()
    in_dim = all_feat_matrix.shape[1]
    device = get_device()
    model = SSQTCNModel(input_dim=in_dim).to(device)
    model_path = "data/tcn_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    f_engine = SSQFilter(cfg.get("filters", {}))
    top_n = cfg.get("sampler", {}).get("top_n_final", 5)
    all_feat_matrix = extractor.get_feature_matrix()

    # 逐期生成预测
    simulated_preds = []
    # 待评测的开奖是最后 test_issues 期
    total_len = len(extractor.records)
    start_eval_idx = total_len - test_issues

    for eval_idx in range(start_eval_idx, total_len):
        # 严格只使用 [eval_idx - w_size, eval_idx) 的特征
        hist_feats = all_feat_matrix[eval_idx - w_size: eval_idx]
        with torch.no_grad():
            x_t = torch.tensor(hist_feats, dtype=torch.float32).unsqueeze(0).to(device)
            r_l, b_l = model(x_t)
            r_p = torch.sigmoid(r_l).squeeze(0).cpu().numpy()
            b_p = torch.softmax(b_l, dim=-1).squeeze(0).cpu().numpy()

        sampler = SSQSampler(r_p, b_p, tau=1.0)
        cands = sampler.sample_candidates(n_candidates=3000, random_seed=eval_idx)
        passed, _ = f_engine.filter_candidates(cands)
        if len(passed) < top_n:
            passed = cands  # 保底
        chosen = f_engine.select_diverse_tickets(passed, top_n=top_n)
        simulated_preds.append(chosen)

    # 启动回测分析器
    backtester = SSQBacktester(extractor.records, test_issues=test_issues, config=cfg)
    results = backtester.run_simulation(simulated_preds)
    backtester.generate_report_chart(results, save_path="reports/backtest_report.png")

    # 保存摘要 JSON
    with open("reports/backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"         📊 【历史 {test_issues} 期 Walk-Forward 回测全面报告】")
    print("=" * 65)
    print(f"  • 总评估期数: {results['test_issues']} 期 | 每期选号: {results['tickets_per_issue']} 注 (总计 {results['total_tickets_evaluated']} 注)")
    print(f"  • 总购彩投入: {results['total_cost_yuan']:.1f} 元 | 累计中奖返还: {results['total_payout_yuan']:.1f} 元")
    print(f"  • 累计净损益: {results['total_profit_yuan']:.1f} 元 | 实际返还率 (ROI): {results['roi_percent']:.2f}%")
    print(f"  • 红球平均命中: {results['red_hit_avg']:.4f} 个 (理论随机期望: {results['red_hit_theoretical']:.4f} 个)")
    print(f"  • 蓝球命中概率: {results['blue_hit_rate']*100:.2f}% (理论随机期望: {results['blue_hit_theoretical']*100:.2f}%)")
    print(f"  • 各奖级明细: {results['prize_counter']}")
    print(f"  • 统计显著性检验: {'⚠️ 显著偏离随机' if results['is_significant'] else '✅ 落在随机 95% 置信区间内 (符合无超额收益科学预期)'}")
    print("=" * 65)
    print(DISCLAIMER)


def cmd_train_stgnn(args, cfg):
    import csv
    from src.model_stgnn import SSQComplexSTGNN, build_domain_adj_matrices
    from src.dataset_stgnn import SSQGraphVisualDataset

    csv_path = cfg.get("data", {}).get("csv_path", "data/raw/ssq_history.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    w_size = cfg.get("features", {}).get("window_size", 20)
    A_domain, A_co, A_trans = build_domain_adj_matrices(records)
    dataset = SSQGraphVisualDataset(records, window_size=w_size)

    test_size = 100
    train_size = len(dataset) - test_size
    train_indices = list(range(train_size))
    test_indices = list(range(train_size, len(dataset)))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    batch_size = cfg.get("model", {}).get("batch_size", 32)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    device = get_device()
    print(f"[*] 训练复杂时空图网络 ST-GNN-Vision (设备: {device})")

    model = SSQComplexSTGNN(
        node_in_feats=6,
        hidden_dim=64,
        num_layers=2,
        window_size=w_size,
        adj_domain=A_domain,
        adj_co=A_co,
        adj_trans=A_trans,
        dropout=0.2
    ).to(device)

    epochs = args.epochs or 25
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    print(f"[*] 开始训练 ST-GNN-Vision 模型 (共 {epochs} 轮)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for n_seq, t_grid, b_hist, y_red, y_blue in train_loader:
            n_seq = n_seq.to(device)
            t_grid = t_grid.to(device)
            b_hist = b_hist.to(device)
            y_red = y_red.to(device)
            y_blue = y_blue.to(device)

            optimizer.zero_grad()
            r_logits, b_logits = model(n_seq, t_grid, b_hist)
            loss, _, _ = compute_loss(r_logits, b_logits, y_red, y_blue)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(n_seq)

        avg_train = total_loss / train_size
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for n_seq, t_grid, b_hist, y_red, y_blue in test_loader:
                    n_seq, t_grid, b_hist = n_seq.to(device), t_grid.to(device), b_hist.to(device)
                    y_red, y_blue = y_red.to(device), y_blue.to(device)
                    r_logits, b_logits = model(n_seq, t_grid, b_hist)
                    loss, _, _ = compute_loss(r_logits, b_logits, y_red, y_blue)
                    total_val += loss.item() * len(n_seq)
            avg_val = total_val / test_size
            print(f"  ST-GNN Epoch [{epoch:02d}/{epochs:02d}] - 训练损失: {avg_train:.4f} | 验证损失: {avg_val:.4f}")

    os.makedirs("data", exist_ok=True)
    save_path = "data/stgnn_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[+] 复杂时空图模型已保存至: {save_path}")


def cmd_predict_stgnn(args, cfg):
    import csv
    from src.model_stgnn import SSQComplexSTGNN, build_domain_adj_matrices
    from src.dataset_stgnn import SSQGraphVisualDataset

    csv_path = cfg.get("data", {}).get("csv_path", "data/raw/ssq_history.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    w_size = cfg.get("features", {}).get("window_size", 20)
    A_domain, A_co, A_trans = build_domain_adj_matrices(records)
    dataset = SSQGraphVisualDataset(records, window_size=w_size)

    device = get_device()
    model = SSQComplexSTGNN(
        node_in_feats=6, hidden_dim=64, num_layers=2, window_size=w_size,
        adj_domain=A_domain, adj_co=A_co, adj_trans=A_trans
    ).to(device)

    model_path = "data/stgnn_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[*] 已载入预训练 ST-GNN-Vision 权重: {model_path}")
    else:
        print("[!] 未找到 ST-GNN 权重，请先执行: python main.py train_stgnn")
        return

    # 提取最后一组样本
    n_seq, t_grid, b_hist, _, _ = dataset[len(dataset) - 1]
    model.eval()
    with torch.no_grad():
        r_l, b_l = model(n_seq.unsqueeze(0).to(device), t_grid.unsqueeze(0).to(device), b_hist.unsqueeze(0).to(device))
        r_probs = torch.sigmoid(r_l).squeeze(0).cpu().numpy()
        b_probs = torch.softmax(b_l, dim=-1).squeeze(0).cpu().numpy()

    sampler = SSQSampler(r_probs, b_probs, tau=1.0)
    cands = sampler.sample_candidates(n_candidates=8000)
    f_engine = SSQFilter(cfg.get("filters", {}))
    passed, stats = f_engine.filter_candidates(cands)
    print(f"[*] ST-GNN 走势图采样 {stats['total_input']} 注 -> 过滤合格 {stats['passed_count']} 注")

    top_n = args.num or 5
    selected = f_engine.select_diverse_tickets(passed, top_n=top_n)

    print("\n" + "=" * 65)
    print(f"  🌌 【ST-GNN-Vision 多拓扑时空图网络精选 {len(selected)} 注推荐】")
    print("=" * 65)
    for idx, t in enumerate(selected, 1):
        red_str = " ".join(f"{x:02d}" for x in t["reds"])
        blue_str = f"{t['blue']:02d}"
        s_red = sum(t["reds"])
        odds = sum(1 for x in t["reds"] if x % 2 == 1)
        print(f"  第 {idx} 注:  🔴 红球 [ {red_str} ]  🔵 蓝球 [ {blue_str} ]  | 和值: {s_red:03d} 奇偶: {odds}:{6-odds}")
    print("=" * 65)
    print(DISCLAIMER)


def cmd_backtest_stgnn(args, cfg):
    import csv
    from src.model_stgnn import SSQComplexSTGNN, build_domain_adj_matrices
    from src.dataset_stgnn import SSQGraphVisualDataset

    csv_path = cfg.get("data", {}).get("csv_path", "data/raw/ssq_history.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    w_size = cfg.get("features", {}).get("window_size", 20)
    test_issues = args.issues or cfg.get("backtest", {}).get("test_issues", 100)
    print(f"[*] 启动 ST-GNN-Vision 复杂时空图对过去 {test_issues} 期的 Walk-Forward 滚动前向回测...")

    A_domain, A_co, A_trans = build_domain_adj_matrices(records)
    dataset = SSQGraphVisualDataset(records, window_size=w_size)

    device = get_device()
    model = SSQComplexSTGNN(
        node_in_feats=6, hidden_dim=64, num_layers=2, window_size=w_size,
        adj_domain=A_domain, adj_co=A_co, adj_trans=A_trans
    ).to(device)

    model_path = "data/stgnn_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[*] 已载入预训练 ST-GNN-Vision 权重: {model_path}")
    else:
        print("[!] 未找到 ST-GNN 权重，先执行训练...")
        cmd_train_stgnn(args, cfg)

    model.eval()
    f_engine = SSQFilter(cfg.get("filters", {}))
    top_n = cfg.get("sampler", {}).get("top_n_final", 5)

    simulated_preds = []
    total_len = len(dataset)
    start_eval_idx = total_len - test_issues

    for eval_idx in range(start_eval_idx, total_len):
        n_seq, t_grid, b_hist, _, _ = dataset[eval_idx]
        with torch.no_grad():
            r_l, b_l = model(
                n_seq.unsqueeze(0).to(device),
                t_grid.unsqueeze(0).to(device),
                b_hist.unsqueeze(0).to(device)
            )
            r_p = torch.sigmoid(r_l).squeeze(0).cpu().numpy()
            b_p = torch.softmax(b_l, dim=-1).squeeze(0).cpu().numpy()

        sampler = SSQSampler(r_p, b_p, tau=1.0)
        cands = sampler.sample_candidates(n_candidates=3000, random_seed=eval_idx)
        passed, _ = f_engine.filter_candidates(cands)
        if len(passed) < top_n:
            passed = cands
        chosen = f_engine.select_diverse_tickets(passed, top_n=top_n)
        simulated_preds.append(chosen)

    backtester = SSQBacktester(records, test_issues=test_issues, config=cfg)
    results = backtester.run_simulation(simulated_preds)
    backtester.generate_report_chart(results, save_path="reports/backtest_stgnn_report.png")

    with open("reports/backtest_stgnn_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"    🌌 【ST-GNN-Vision 历史 {test_issues} 期 Walk-Forward 回测全面报告】")
    print("=" * 65)
    print(f"  • 总评估期数: {results['test_issues']} 期 | 每期选号: {results['tickets_per_issue']} 注 (总计 {results['total_tickets_evaluated']} 注)")
    print(f"  • 总购彩投入: {results['total_cost_yuan']:.1f} 元 | 累计中奖返还: {results['total_payout_yuan']:.1f} 元")
    print(f"  • 累计净损益: {results['total_profit_yuan']:.1f} 元 | 实际返还率 (ROI): {results['roi_percent']:.2f}%")
    print(f"  • 红球平均命中: {results['red_hit_avg']:.4f} 个 (理论随机期望: {results['red_hit_theoretical']:.4f} 个)")
    print(f"  • 蓝球命中概率: {results['blue_hit_rate']*100:.2f}% (理论随机期望: {results['blue_hit_theoretical']*100:.2f}%)")
    print(f"  • 各奖级明细: {results['prize_counter']}")
    print(f"  • 统计显著性检验: {'⚠️ 显著偏离随机' if results['is_significant'] else '✅ 落在随机 95% 置信区间内 (符合无超额收益科学预期)'}")
    print("=" * 65)
    print(DISCLAIMER)


def main():
    parser = argparse.ArgumentParser(description="SSQ-Prediction-AI: 双色球 AI 选号与科学实验系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # fetch
    subparsers.add_parser("fetch", help="抓取全量数据入库并进行统计学检验")

    # train
    train_parser = subparsers.add_parser("train", help="训练 TCN 深度学习模型")
    train_parser.add_argument("--epochs", type=int, default=None, help="训练轮数")

    # predict
    predict_parser = subparsers.add_parser("predict", help="执行 AI 采样与智能缩水推荐")
    predict_parser.add_argument("--num", type=int, default=5, help="精选推荐注数")

    # backtest
    backtest_parser = subparsers.add_parser("backtest", help="执行滚动时序回测对比随机基准")
    backtest_parser.add_argument("--issues", type=int, default=100, help="回测期数")

    # train_stgnn
    stgnn_train_parser = subparsers.add_parser("train_stgnn", help="训练复杂时空图与走势图双流模型 (ST-GNN-Vision)")
    stgnn_train_parser.add_argument("--epochs", type=int, default=25, help="训练轮数")

    # predict_stgnn
    stgnn_pred_parser = subparsers.add_parser("predict_stgnn", help="使用 ST-GNN-Vision 生成推荐号码")
    stgnn_pred_parser.add_argument("--num", type=int, default=5, help="精选推荐注数")

    # backtest_stgnn
    stgnn_bt_parser = subparsers.add_parser("backtest_stgnn", help="执行 ST-GNN-Vision 滚动时序回测")
    stgnn_bt_parser.add_argument("--issues", type=int, default=100, help="回测期数")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "fetch":
        cmd_fetch(args, cfg)
    elif args.command == "train":
        cmd_train(args, cfg)
    elif args.command == "predict":
        cmd_predict(args, cfg)
    elif args.command == "backtest":
        cmd_backtest(args, cfg)
    elif args.command == "train_stgnn":
        cmd_train_stgnn(args, cfg)
    elif args.command == "predict_stgnn":
        cmd_predict_stgnn(args, cfg)
    elif args.command == "backtest_stgnn":
        cmd_backtest_stgnn(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
