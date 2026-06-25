#!/usr/bin/env bash
# ============================================================================
# Solar-Energy 训练 (Table 1 长预测: pred_len 96/192/336/720, 137 变量)
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_Solar.sh            # 跑全部 4 个 pred_len
#   bash scripts/reproduce/run_Solar.sh 96 192     # 只跑指定 pred_len
#   GPU=1 bash scripts/reproduce/run_Solar.sh      # 用 1 号卡
# 数据: ./dataset/Solar/solar_AL.txt   (注意: 目录 Solar 首字母大写; 文件 .txt)
# 结果: results/solar_96_*/metrics.npy + result_long_term_forecast.txt
# 参数: --data Solar(专用 Dataset_Solar 加载器), e_layers=2, d_model=d_ff=512, lr=0.0005,
#       batch_size=32(run.py 默认, 官方未显式指定)
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PLS="${*:-96 192 336 720}"
echo "============================================================"
echo " Solar-Energy 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- Solar pred_len=$pl ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id solar_96_${pl} \
    --model iTransformer \
    --data Solar \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers 2 \
    --enc_in 137 --dec_in 137 --c_out 137 \
    --d_model 512 --d_ff 512 \
    --learning_rate 0.0005 \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " Solar-Energy 训练完成 | 结果见 results/solar_96_*"
echo "============================================================"
