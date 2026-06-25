#!/usr/bin/env python
"""
extract_results.py — 从 result_long_term_forecast.txt 提取 MSE/MAE，生成 §3 进度表 Markdown

用法：
    python extract_results.py                                    # 自动找项目根目录下的文件
    python extract_results.py /path/to/result_long_term_forecast.txt  # 指定路径

输出到 stdout，重定向保存：
    python extract_results.py > section3.md

在服务器上跑完后把输出的 markdown 复制回来，替换 实验记录.md 的 §3 即可。
"""

import argparse
import io
import os
import re
import sys

# 保障 stdout 能输出 UTF-8（Windows GBK 终端 + emoji/SJIS 字符）
if sys.stdout.encoding and sys.stdout.encoding.upper() != "UTF-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================
# 论文 Table 1 参考值（严格复现对照基准）
# 格式: {显示名: {pred_len: (论文MSE, 论文MAE)}}
# ============================================================
PAPER = {
    "ETTh1":          {96: (0.386, 0.405), 192: (0.451, 0.445), 336: (0.432, 0.433), 720: (0.431, 0.435)},
    "ETTh2":          {96: (0.297, 0.348), 192: (0.383, 0.407), 336: (0.400, 0.428), 720: (0.427, 0.448)},
    "ETTm1":          {96: (0.334, 0.370), 192: (0.377, 0.397), 336: (0.426, 0.435), 720: (0.446, 0.449)},
    "ETTm2":          {96: (0.180, 0.264), 192: (0.251, 0.321), 336: (0.314, 0.360), 720: (0.371, 0.396)},
    "Weather":        {96: (0.164, 0.210), 192: (0.209, 0.254), 336: (0.260, 0.296), 720: (0.325, 0.344)},
    "Electricity(ECL)": {96: (0.140, 0.237), 192: (0.153, 0.248), 336: (0.168, 0.267), 720: (0.208, 0.298)},
    "Traffic":        {96: (0.395, 0.268), 192: (0.419, 0.287), 336: (0.444, 0.309), 720: (0.471, 0.329)},
}

# setting 前缀 → 显示名（按显示顺序排列）
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

# 偏差判定阈值
OK_THRESHOLD = 8.0     # MSE 偏差 < 8% → ✅
WARN_THRESHOLD = 13.0  # MSE 偏差 ≥ 13% → ⚠️


def parse_setting(setting: str):
    """从 setting 行解析出 (显示名, pred_len)，非实验或无效则返回 None。"""
    if setting.startswith("smoke_"):
        return None

    for prefix, display_name in PREFIXES:
        if setting.startswith(prefix + "_"):
            # 取前缀后的第一段（seq_len）和第二段（pred_len）
            rest = setting[len(prefix) + 1:]
            parts = rest.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                pred_len = int(parts[1])
                if pred_len in PRED_LENS:
                    return display_name, pred_len
    return None


def calc_dev(mse, paper_mse):
    """MSE 偏差百分比"""
    return (mse - paper_mse) / paper_mse * 100 if paper_mse else 0


def status(dev):
    if dev < OK_THRESHOLD:
        return "✅"
    elif dev >= WARN_THRESHOLD:
        return "⚠️"
    else:
        return "🔶"


def cell_text(mse, mae, dev):
    return f"{status(dev)} {mse:.3f} / {mae:.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="从 result_long_term_forecast.txt 提取 MSE/MAE 生成进度表 Markdown"
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="result_long_term_forecast.txt 路径（默认自动在项目根目录查找）")
    args = parser.parse_args()

    # 定位文件
    txt = args.input
    if not txt:
        # 脚本在 scripts/reproduce/，项目根是 ../../
        guess = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "result_long_term_forecast.txt")
        txt = os.path.normpath(guess)

    if not os.path.isfile(txt):
        print(f"[错误] 找不到 {txt}", file=sys.stderr)
        sys.exit(1)

    # 解析
    results = {}  # {显示名: {pred_len: (mse, mae)}}
    with open(txt, "r", encoding="utf-8") as f:
        text = f.read()

    # 按行处理，跳过 smoke 条目
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("smoke_"):
            i += 1
            continue

        parsed = parse_setting(line)
        if parsed is None:
            i += 1
            continue

        ds_name, pred_len = parsed

        # 下一行必须是 mse:..., mae:...
        if i + 1 >= len(lines):
            break
        m = re.match(r"mse:([0-9.eE+\-]+),\s*mae:([0-9.eE+\-]+)", lines[i + 1].strip())
        if not m:
            i += 1
            continue

        mse, mae = float(m.group(1)), float(m.group(2))

        results.setdefault(ds_name, {})[pred_len] = (mse, mae)
        i += 2

    # ---- 输出 ----
    total = len(PREFIXES) * len(PRED_LENS)
    done = sum(1 for ds in results.values() for pl in ds if pl in PRED_LENS)

    print("## 3. 进度总览（7 × 4，均按官方默认参数「严格复现」）")
    print()
    print("格内为复现 **MSE / MAE**。符号：✅ 复现成功（偏差 <8%）｜🔶 偏差略大（8%~13%）｜"
          "⚠️ 偏差大（>13%，待调参）｜❌ 未开始")
    print()
    print("| 数据集 | pred_len 96 | 192 | 336 | 720 |")
    print("|---|---|---|---|---|")

    for prefix, ds_name in PREFIXES:
        row = f"| **{ds_name}**"
        for pl in PRED_LENS:
            if ds_name in results and pl in results[ds_name]:
                mse, mae = results[ds_name][pl]
                if ds_name in PAPER and pl in PAPER[ds_name]:
                    pmse, _ = PAPER[ds_name][pl]
                    row += f" | {cell_text(mse, mae, calc_dev(mse, pmse))}"
                else:
                    row += f" | 📋 {mse:.3f} / {mae:.3f}"
            else:
                row += " | ❌"
        row += " |"
        print(row)

    print()
    print(f"**完成度**：{done} / {total} 项（{done / total * 100:.0f}%）。")
    print()

    # ---- §4 详细结果 ----
    print("---")
    print()
    print("## 4. 已完成结果（严格默认参数）")
    print()

    for idx, (prefix, ds_name) in enumerate([p for p in PREFIXES if p[1] in results], 1):
        print(f"### 4.{idx} {ds_name}")
        print()
        print("| pred_len | 复现 MSE | 复现 MAE | 论文 MSE | 论文 MAE | MSE 偏差 | 判定 |")
        print("|---|---|---|---|---|---|---|")

        for pl in PRED_LENS:
            if ds_name in results and pl in results[ds_name]:
                mse, mae = results[ds_name][pl]
                if ds_name in PAPER and pl in PAPER[ds_name]:
                    pmse, pmae = PAPER[ds_name][pl]
                    dev = calc_dev(mse, pmse)
                    print(f"| {pl} | {mse:.4f} | {mae:.4f} | {pmse:.3f} | {pmae:.3f} | {dev:+.1f}% | {status(dev)} |")
                else:
                    print(f"| {pl} | {mse:.4f} | {mae:.4f} | — | — | — | 📋 |")
            else:
                print(f"| {pl} | — | — | — | — | — | ❌ |")
        print()


if __name__ == "__main__":
    main()
