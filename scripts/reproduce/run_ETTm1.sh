#!/usr/bin/env bash
# ============================================================================
# ETTm1 训练 (Table 1 主表: pred_len 96/192/336/720)
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_ETTm1.sh            # 跑全部 4 个 pred_len
#   bash scripts/reproduce/run_ETTm1.sh 96 192     # 只跑指定 pred_len
#   GPU=1 bash scripts/reproduce/run_ETTm1.sh      # 用 1 号卡
# 结果: results/ETTm1_96_*/metrics.npy + result_long_term_forecast.txt
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PLS="${*:-96 192 336 720}"
echo "============================================================"
echo " ETTm1 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- ETTm1 pred_len=$pl ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_96_${pl} \
    --model iTransformer \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers 2 \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model 128 --d_ff 128 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " ETTm1 训练完成 | 结果见 results/ETTm1_96_*"
echo "============================================================"
