#!/usr/bin/env python
"""
extract_results_tune.py — 从 result_long_term_forecast.txt 按 des 分组提取 MSE/MAE，
生成第二阶段调参对比 Markdown（用于 实验记录.md §7.2）。

与 extract_results_full.py 的区别:
  - parse_setting 额外解析 des（setting 末尾倒数第三段），不丢弃；
  - 结果按 des 分组 results[des][ds][pred_len]，避免"最后一条覆盖"；
  - 输出按 des 分节，每个 tune 方案一节，含 vs论文偏差 + vs默认(Exp)改善。

用法:
    python scripts/reproduce/extract_results_tune.py
    python scripts/reproduce/extract_results_tune.py /path/to/result_long_term_forecast.txt
    python scripts/reproduce/extract_results_tune.py > section7_2.md

约定: des 不含下划线（如 Exp / TuneC20p5 / TuneE20p5），否则倒数第三段解析会错。
前置: 调参前已 cp result_long_term_forecast.txt result_long_term_forecast.default.txt。
注意: PAPER 仅含 7 个长预测数据集（ETTh1/ETTh2/ETTm1/ETTm2/Weather/ECL/Traffic），
      本次调参目标（ETTh1/ETTm2/Weather/ECL）均在内；若需 Exchange/Solar 请从
      extract_results_full.py 复制 PAPER。
"""

import argparse
import io
import os
import re
import sys

# 保障 stdout 输出 UTF-8（Windows GBK 终端 + emoji）
if sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================
# 论文 Table 10 真实值（与 extract_results_full.py 一致；已对照 arXiv v4 订正 2026-06-30）
# ============================================================
PAPER = {
    "ETTh1":            {96: (0.386, 0.405), 192: (0.441, 0.436), 336: (0.487, 0.458), 720: (0.503, 0.491)},
    "ETTh2":            {96: (0.297, 0.349), 192: (0.380, 0.400), 336: (0.428, 0.432), 720: (0.427, 0.445)},
    "ETTm1":            {96: (0.334, 0.368), 192: (0.377, 0.391), 336: (0.426, 0.420), 720: (0.491, 0.459)},
    "ETTm2":            {96: (0.180, 0.264), 192: (0.250, 0.309), 336: (0.311, 0.348), 720: (0.412, 0.407)},
    "Weather":          {96: (0.174, 0.214), 192: (0.221, 0.254), 336: (0.278, 0.296), 720: (0.358, 0.347)},
    "Electricity(ECL)": {96: (0.148, 0.240), 192: (0.162, 0.253), 336: (0.178, 0.269), 720: (0.225, 0.317)},
    "Traffic":          {96: (0.395, 0.268), 192: (0.417, 0.276), 336: (0.433, 0.283), 720: (0.467, 0.302)},
}

# setting 前缀 → 显示名（按显示顺序）
PREFIXES = [
    ("ETTh1", "ETTh1"),
    ("ETTh2", "ETTh2"),
    ("ETTm1", "ETTm1"),
    ("ETTm2", "ETTm2"),
    ("weather", "Weather"),
    ("ECL", "Electricity(ECL)"),
    ("traffic", "Traffic"),
]

PRED_LENS = [96, 192, 336, 720]

# 调参成功线（比第一阶段 8%/13% 更严，目标论文级）
OK_THRESHOLD = 5.0      # MSE 偏差 < 5% → ✅ 达标
WARN_THRESHOLD = 8.0    # MSE 偏差 ≥ 8% → ⚠️ 仍未达标

DEFAULT_DES = "Exp"     # 第一阶段默认参数的 des（基线）


def parse_setting(setting: str):
    """返回 (显示名, pred_len, des)；非实验或无效返回 None。"""
    if setting.startswith("smoke_"):
        return None
    for prefix, display_name in PREFIXES:
        if setting.startswith(prefix + "_"):
            rest = setting[len(prefix) + 1:]
            parts = rest.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                pred_len = int(parts[1])
                if pred_len in PRED_LENS:
                    # setting 末尾: ..._dt{distil}_{des}_{class_strategy}_{itr}
                    # des = 倒数第三段
                    tail = setting.split("_")
                    des = tail[-3] if len(tail) >= 3 else "?"
                    return display_name, pred_len, des
    return None


def calc_dev(mse, paper_mse):
    return (mse - paper_mse) / paper_mse * 100 if paper_mse else 0


def status(dev):
    if dev is None:
        return "—"
    if dev < OK_THRESHOLD:
        return "✅"
    elif dev >= WARN_THRESHOLD:
        return "⚠️"
    else:
        return "🔶"


def main():
    parser = argparse.ArgumentParser(description="按 des 分组提取调参对比表")
    parser.add_argument("input", nargs="?", default=None,
                        help="result_long_term_forecast.txt 路径（默认项目根目录）")
    args = parser.parse_args()

    txt = args.input
    if not txt:
        guess = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "result_long_term_forecast.txt")
        txt = os.path.normpath(guess)
    if not os.path.isfile(txt):
        print(f"[错误] 找不到 {txt}", file=sys.stderr)
        sys.exit(1)

    # results[des][display_name][pred_len] = (mse, mae)
    results = {}
    with open(txt, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("smoke_"):
            i += 1
            continue
        parsed = parse_setting(line)
        if parsed is None:
            i += 1
            continue
        ds_name, pred_len, des = parsed
        if i + 1 >= len(lines):
            break
        m = re.match(r"mse:([0-9.eE+\-]+),\s*mae:([0-9.eE+\-]+)", lines[i + 1].strip())
        if not m:
            i += 1
            continue
        mse, mae = float(m.group(1)), float(m.group(2))
        results.setdefault(des, {}).setdefault(ds_name, {})[pred_len] = (mse, mae)
        i += 2

    baseline = results.get(DEFAULT_DES, {})
    des_list = sorted(results.keys())

    print("## 7.2 长预测调参对比（按 des 分组）")
    print()
    print(f"数据源: `{os.path.basename(txt)}` ｜ 成功线: MSE 偏差 < {OK_THRESHOLD:.0f}%（论文级）")
    print()
    print(f"**检出 des**: {', '.join(des_list) if des_list else '（无）'}")
    print()
    print("> 最佳 epoch / 训练总轮数需从 `logs/tune_*.log` 手动补入；此表仅含指标。")
    print()

    for des in des_list:
        grp = results[des]
        is_default = (des == DEFAULT_DES)
        print("---")
        print()
        title = f"{des}（默认基线）" if is_default else f"{des}（调参）"
        print(f"### des = {title}")
        print()

        if is_default:
            print("| 数据集/pred_len | 复现 MSE | 复现 MAE | 论文 MSE | 论文 MAE | vs 论文偏差 | 判定 |")
            print("|---|---|---|---|---|---|---|")
        else:
            print("| 数据集/pred_len | 复现 MSE | 复现 MAE | 论文 MSE | vs 论文偏差 | vs 默认(Exp) | 判定 |")
            print("|---|---|---|---|---|---|---|")

        rows = 0
        for _, ds_name in PREFIXES:
            if ds_name not in grp:
                continue
            for pl in PRED_LENS:
                if pl not in grp[ds_name]:
                    continue
                mse, mae = grp[ds_name][pl]
                pmse = pmae = None
                if ds_name in PAPER and pl in PAPER[ds_name]:
                    pmse, pmae = PAPER[ds_name][pl]
                dev = calc_dev(mse, pmse) if pmse else None
                dev_s = f"{dev:+.1f}%" if dev is not None else "—"
                pmse_s = f"{pmse:.3f}" if pmse is not None else "—"

                if is_default:
                    pmae_s = f"{pmae:.3f}" if pmae is not None else "—"
                    print(f"| {ds_name}/{pl} | {mse:.4f} | {mae:.4f} | {pmse_s} | {pmae_s} | {dev_s} | {status(dev)} |")
                else:
                    if ds_name in baseline and pl in baseline[ds_name]:
                        base_mse = baseline[ds_name][pl][0]
                        improve = (base_mse - mse) / base_mse * 100
                        imp_s = f"{improve:+.1f}%"
                    else:
                        imp_s = "—"
                    print(f"| {ds_name}/{pl} | {mse:.4f} | {mae:.4f} | {pmse_s} | {dev_s} | {imp_s} | {status(dev)} |")
                rows += 1
        if rows == 0:
            print("| — | — | — | — | — | — | — |")
        print()
        if not is_default:
            print(f"> `vs 默认(Exp)` = (Exp_MSE − {des}_MSE) / Exp_MSE × 100%，正值 = 变好。")
            print()


if __name__ == "__main__":
    main()
