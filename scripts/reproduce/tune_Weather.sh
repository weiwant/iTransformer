#!/usr/bin/env bash
# ============================================================================
# Weather 调参（第二阶段，针对系统性偏差项 pred_len 720，21 变量）
# 与默认参数(run_Weather.sh, --des Exp)的区别: --train_epochs/--patience/--lradj + --des 隔离。
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/tune_Weather.sh            # 跑 720
#   GPU=1 bash scripts/reproduce/tune_Weather.sh      # 用 1 号卡
#   开发机自测: PY="D:/SoftWare22/miniconda/envs/itransformer/python.exe" bash scripts/reproduce/tune_Weather.sh 720
# 消融: 只改下方 4 个旋钮即跑单变量对照。
# 前置: 先备份 result_long_term_forecast.txt（见 tune_ETTh1.sh 注释）。
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=${PY:-python}

# ---- 调参旋钮（均支持环境变量覆盖；消融时用 env 覆盖，如 LRADJ=type1 DES=TuneE20p5） ----
EPOCHS=${EPOCHS:-20}
PATIENCE=${PATIENCE:-5}
LRADJ=${LRADJ:-cosine}
DES=${DES:-TuneC20p5}

PLS="${*:-720}"
mkdir -p logs
echo "============================================================"
echo " Weather 调参 | pred_len=[$PLS] | ep=$EPOCHS pa=$PATIENCE lradj=$LRADJ des=$DES | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- Weather pred_len=$pl ----"
  "$PY" -u run.py --is_training 1 \
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
    --train_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --lradj "$LRADJ" \
    --des "$DES" \
    --itr 1 \
    2>&1 | tee logs/tune_Weather_${DES}_pl${pl}.log
done

echo "============================================================"
echo " Weather 调参完成 | des=$DES | 结果见 result_long_term_forecast.txt 与 logs/"
echo "============================================================"
