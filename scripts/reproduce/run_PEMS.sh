#!/usr/bin/env bash
# ============================================================================
# PEMS 训练 (短预测: pred_len 12/24/48/96; 4 个子集 03/04/07/08)
#   注意: 与其余长预测数据集(pred_len 96/192/336/720)不同, PEMS 是短期交通流量预测.
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/run_PEMS.sh                 # 默认 PEMS04, 全部 4 个 pred_len
#   bash scripts/reproduce/run_PEMS.sh 04              # PEMS04 全部
#   bash scripts/reproduce/run_PEMS.sh 08 12 24        # PEMS08 只跑 12/24
#   bash scripts/reproduce/run_PEMS.sh 03 12 24 48 96  # PEMS03 指定 pred_len
#   GPU=1 bash scripts/reproduce/run_PEMS.sh 04        # 1 号卡
# 数据: ./dataset/PEMS/PEMS0X.npz   (PEMS03.npz / PEMS04.npz / PEMS07.npz / PEMS08.npz)
# 结果: results/PEMS0X_96_*/metrics.npy + result_long_term_forecast.txt
# 参数来源: 官方 scripts/multivariate_forecasting/PEMS/iTransformer_0X.sh
#   各子集/各 pred_len 的 e_layers/batch/lr/use_norm 不同, 已内置查表(pems_params).
#   PEMS03: enc_in=358, e_layers=4, d_model=512,  lr=0.001, bs=32, use_norm=1 (全 pred_len)
#   PEMS04: enc_in=307, e_layers=4, d_model=1024, lr=0.0005, bs=32, use_norm=0 (全 pred_len)
#   PEMS07: enc_in=883, d_model=512, lr=0.001, use_norm=0;  pl12/24:e_layers=2,bs=32; pl48/96:e_layers=4,bs=16
#   PEMS08: enc_in=170, d_model=512; pl12/24:e_layers=2,lr=0.0005,bs=32,use_norm=1; pl48/96:e_layers=4,lr=0.001,bs=16,use_norm=0
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}

# 子集号 + pred_len → "enc_in e_layers d_model d_ff lr batch use_norm"
pems_params() {
  local s="$1" pl="$2"
  case "$s" in
    03) echo "358 4 512 512 0.001 32 1" ;;
    04) echo "307 4 1024 1024 0.0005 32 0" ;;
    07) case "$pl" in
          12|24) echo "883 2 512 512 0.001 32 0" ;;
          48|96) echo "883 4 512 512 0.001 16 0" ;;
        esac ;;
    08) case "$pl" in
          12|24) echo "170 2 512 512 0.0005 32 1" ;;
          48|96) echo "170 4 512 512 0.001 16 0" ;;
        esac ;;
    *) echo "未知子集 PEMS${s} (支持: 03 04 07 08)" >&2; exit 1 ;;
  esac
}

SUBSET="${1:-04}"
case "$SUBSET" in
  03|04|07|08) shift ;;
  *) echo "用法: $0 [03|04|07|08] [pred_len...]" >&2; exit 1 ;;
esac

PLS="${*:-12 24 48 96}"
echo "============================================================"
echo " PEMS${SUBSET} 训练 | pred_len=[$PLS] | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  params="$(pems_params "$SUBSET" "$pl")"
  if [[ -z "$params" ]]; then
    echo "PEMS${SUBSET} 不支持 pred_len=$pl (可用: 12 24 48 96), 跳过" >&2
    continue
  fi
  read -r enc elayers dm dff lr bs un <<< "$params"
  echo "---- PEMS${SUBSET} pred_len=$pl (enc_in=$enc e_layers=$elayers d_model=$dm lr=$lr bs=$bs use_norm=$un) ----"
  python -u run.py --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS${SUBSET}.npz \
    --model_id PEMS${SUBSET}_96_${pl} \
    --model iTransformer \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len "$pl" \
    --e_layers "$elayers" \
    --enc_in "$enc" --dec_in "$enc" --c_out "$enc" \
    --d_model "$dm" --d_ff "$dff" \
    --batch_size "$bs" \
    --learning_rate "$lr" \
    --use_norm "$un" \
    --des Exp \
    --itr 1
done

echo "============================================================"
echo " PEMS${SUBSET} 训练完成 | 结果见 results/PEMS${SUBSET}_96_*"
echo "============================================================"
