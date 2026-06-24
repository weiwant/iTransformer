#!/usr/bin/env bash
# ============================================================================
# iTransformer 复现 — 冒烟测试 (smoke test)
# ----------------------------------------------------------------------------
# 用途: 用最小配置 (ETTh1, pred_len=96, train_epochs=1) 快速验证环境能否跑通
#       iTransformer 全流程: 数据加载 -> 训练 -> 测试 -> 指标 -> 保存
#       -> checkpoint 续测。
#
# 用法:
#   conda activate iTransformer        # 先激活环境 (开发机环境名为 itransformer)
#   bash scripts/reproduce/smoke_test.sh
#   GPU=1 bash scripts/reproduce/smoke_test.sh    # 指定使用 1 号 GPU
#
# 预期: 几分钟内跑完, 打印 test MSE/MAE, 生成 checkpoints/results/test_results,
#       末尾打印 "冒烟测试通过".
#
# 注: model_id 用 smoke_ 前缀、--des Smoke, 与正式实验 (--des Exp) 的 setting
#     目录不同, 不会污染正式实验结果。
# ============================================================================

set -euo pipefail

# ---- 配置 ----
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# 定位项目根目录 (本脚本位于 scripts/reproduce/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATA_FILE="./dataset/ETT-small/ETTh1.csv"

echo "============================================================"
echo " iTransformer 冒烟测试 (smoke test)"
echo "============================================================"
echo "[时间] $(date 2>/dev/null || echo '-')"
echo "[GPU ] CUDA_VISIBLE_DEVICES=$GPU"
echo "[目录] $ROOT"
echo ""

# ---- 1. 环境自检 ----
echo "---- [1/4] 环境自检 ----"
python --version
if ! python -c "import torch" 2>/dev/null; then
  echo "[错误] 无法 import torch — 请先激活环境: conda activate iTransformer"
  exit 1
fi
python -c "
import torch
print(f'[torch] {torch.__version__}  cuda_available={torch.cuda.is_available()}  '
      f'device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU (仍可跑, 仅慢)\"}')
"
echo ""

# ---- 2. 数据集存在性检查 ----
echo "---- [2/4] 数据集检查 ----"
if [[ ! -f "$DATA_FILE" ]]; then
  echo "[错误] 数据集不存在: $DATA_FILE"
  echo "       请先下载数据集解压到 dataset/ (见 CLAUDE.md §8, 百度网盘 pwd 9qjr)"
  exit 1
fi
echo "[OK] $DATA_FILE"
echo ""

# ---- 3. 冒烟训练 + 测试 (train_epochs=1) ----
echo "---- [3/4] 训练 + 测试: ETTh1 pred_len=96 train_epochs=1 ----"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id smoke_ETTh1_96_96 \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 256 \
  --train_epochs 1 \
  --des Smoke \
  --itr 1
echo ""

# ---- 训练结果验证 ----
echo "---- 训练结果 ----"
python - <<'PY'
import glob, os, numpy as np
res = sorted(glob.glob('results/smoke_ETTh1_96_96_*'))
ckpt = sorted(glob.glob('checkpoints/smoke_ETTh1_96_96_*'))
assert res, "[错误] 未找到 results/smoke_ETTh1_96_96_* — 训练可能未正常完成"
assert ckpt and os.path.exists(os.path.join(ckpt[0], 'checkpoint.pth')), "[错误] 未找到 checkpoint.pth"
m = np.load(os.path.join(res[0], 'metrics.npy'))  # [mae, mse, rmse, mape, mspe]
print(f"[checkpoint] {ckpt[0]}/checkpoint.pth")
print(f"[结果目录 ] {res[0]}")
print(f"[冒烟指标 ] MSE={m[1]:.4f}  MAE={m[0]:.4f}  (仅 1 epoch, 验证跑通, 非复现数值)")
PY
echo ""

# ---- 4. 续测自检 (--is_training 0, 加载 checkpoint) ----
echo "---- [4/4] 续测自检: --is_training 0 (加载 checkpoint) ----"
python -u run.py \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id smoke_ETTh1_96_96 \
  --model iTransformer \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 256 \
  --train_epochs 1 \
  --des Smoke \
  --itr 1
echo ""

echo "============================================================"
echo " 冒烟测试通过 ✅"
echo "------------------------------------------------------------"
echo " 已验证: 环境(torch/GPU) | 数据集 | 训练 | 测试 |"
echo "         指标保存(metrics.npy) | checkpoint 续测"
echo " 服务器可进入正式复现: bash scripts/multivariate_forecasting/<集>/iTransformer*.sh"
echo "============================================================"
