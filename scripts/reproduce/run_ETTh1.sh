#!/usr/bin/env bash
# ============================================================================
# ETTh1 训练 (Table 1 主表: pred_len 96/192/336/720)
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_ETTh1.sh            # 跑全部 4 个 pred_len
#   bash scripts/reproduce/run_ETTh1.sh 96 192     # 只跑指定 pred_len
#   GPU=1 bash scripts/reproduce/run_ETTh1.sh      # 用 1 号卡
# 结果: results/ETTh1_96_*/metrics.npy + result_long_term_forecast.txt
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PLS="${*:-96 192 336 720}"
echo "============================================================"
echo " ETTh1 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  # ETTh1: pred_len 96/192 用 d_model=256; 336/720 用 d_model=512
  if (( pl <= 192 )); then DM=256; else DM=512; fi
  echo "---- ETTh1 pred_len=$pl d_model=$DM ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_${pl} \
    --model iTransformer \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers 2 \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model "$DM" --d_ff "$DM" \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " ETTh1 训练完成 | 结果见 results/ETTh1_96_*"
echo "============================================================"
