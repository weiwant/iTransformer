#!/usr/bin/env bash
# ============================================================================
# Traffic 训练 (Table 1 主表: pred_len 96/192/336/720, 862 变量, 最慢)
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_Traffic.sh            # 跑全部 4 个 pred_len (较久)
#   bash scripts/reproduce/run_Traffic.sh 720        # 只跑 720 (单跑长预测)
#   GPU=1 bash scripts/reproduce/run_Traffic.sh      # 用 1 号卡
# 结果: results/traffic_96_*/metrics.npy + result_long_term_forecast.txt
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PLS="${*:-96 192 336 720}"
echo "============================================================"
echo " Traffic 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- Traffic pred_len=$pl ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_96_${pl} \
    --model iTransformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers 4 \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --d_model 512 --d_ff 512 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " Traffic 训练完成 | 结果见 results/traffic_96_*"
echo "============================================================"
