#!/usr/bin/env bash
# ============================================================================
# Exchange Rate 训练 (Table 1 长预测: pred_len 96/192/336/720, 8 变量)
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_Exchange.sh            # 跑全部 4 个 pred_len
#   bash scripts/reproduce/run_Exchange.sh 96 192     # 只跑指定 pred_len
#   GPU=1 bash scripts/reproduce/run_Exchange.sh      # 用 1 号卡
# 数据: ./dataset/exchange_rate/exchange_rate.csv
# 结果: results/Exchange_96_*/metrics.npy + result_long_term_forecast.txt
# 参数: --data custom, e_layers=2, d_model=d_ff=128, lr=0.0001(run.py 默认, 官方未显式指定)
# 注意: 官方 Exchange 脚本在 pred_len=336 段含 "--train_epochs 1"(疑为笔误,
#       只训 1 epoch 会令 336 结果异常且与其余 pred_len 不一致)。本脚本统一用
#       默认 train_epochs=10, 避免人为欠拟合(见 CLAUDE.md §6 第二阶段策略)。
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PLS="${*:-96 192 336 720}"
echo "============================================================"
echo " Exchange Rate 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- Exchange pred_len=$pl ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_96_${pl} \
    --model iTransformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers 2 \
    --enc_in 8 --dec_in 8 --c_out 8 \
    --d_model 128 --d_ff 128 \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " Exchange Rate 训练完成 | 结果见 results/Exchange_96_*"
echo "============================================================"
