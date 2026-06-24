#!/usr/bin/env bash
# ============================================================================
# Electricity (ECL) 训练 (Table 1 主表: pred_len 96/192/336/720, 321 变量)
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_ECL.sh            # 跑全部 4 个 pred_len
#   bash scripts/reproduce/run_ECL.sh 96 192     # 只跑指定 pred_len
#   GPU=1 bash scripts/reproduce/run_ECL.sh      # 用 1 号卡
# 结果: results/ECL_96_*/metrics.npy + result_long_term_forecast.txt
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PLS="${*:-96 192 336 720}"
echo "============================================================"
echo " Electricity(ECL) 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- ECL pred_len=$pl ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_96_${pl} \
    --model iTransformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers 3 \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --d_model 512 --d_ff 512 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " Electricity(ECL) 训练完成 | 结果见 results/ECL_96_*"
echo "============================================================"
