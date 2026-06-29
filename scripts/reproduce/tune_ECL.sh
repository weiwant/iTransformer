#!/usr/bin/env bash
# ============================================================================
# Electricity(ECL) 调参（第二阶段，针对系统性偏差项 pred_len 192，321 变量）
# 与默认参数(run_ECL.sh, --des Exp)的区别: --train_epochs/--patience/--lradj + --des 隔离。
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/tune_ECL.sh            # 跑 192
#   GPU=1 bash scripts/reproduce/tune_ECL.sh      # 用 1 号卡
#   开发机自测: PY="D:/SoftWare22/miniconda/envs/itransformer/python.exe" bash scripts/reproduce/tune_ECL.sh 192
# 消融: 只改下方 4 个旋钮即跑单变量对照。
# 注意: ECL 数据集较大(321 变量), 单卡训练较慢, 建议单独占一卡跑。
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

PLS="${*:-192}"
mkdir -p logs
echo "============================================================"
echo " ECL 调参 | pred_len=[$PLS] | ep=$EPOCHS pa=$PATIENCE lradj=$LRADJ des=$DES | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  echo "---- ECL pred_len=$pl ----"
  "$PY" -u run.py --is_training 1 \
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
    --train_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --lradj "$LRADJ" \
    --des "$DES" \
    --itr 1 \
    2>&1 | tee logs/tune_ECL_${DES}_pl${pl}.log
done

echo "============================================================"
echo " ECL 调参完成 | des=$DES | 结果见 result_long_term_forecast.txt 与 logs/"
echo "============================================================"
