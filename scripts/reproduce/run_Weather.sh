#!/usr/bin/env bash
# ============================================================================
# Weather 训练 (Table 1 主表: pred_len 96/192/336/720, 21 变量)
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_Weather.sh            # 跑全部 4 个 pred_len
#   bash scripts/reproduce/run_Weather.sh 96 192     # 只跑指定 pred_len
#   GPU=1 bash scripts/reproduce/run_Weather.sh      # 用 1 号卡
# 结果: results/weather_96_*/metrics.npy + result_long_term_forecast.txt
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PLS="${*:-96 192 336 720}"
echo "============================================================"
echo " Weather 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- Weather pred_len=$pl ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_96_${pl} \
    --model iTransformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers 3 \
    --enc_in 21 --dec_in 21 --c_out 21 \
    --d_model 512 --d_ff 512 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " Weather 训练完成 | 结果见 results/weather_96_*"
echo "============================================================"
