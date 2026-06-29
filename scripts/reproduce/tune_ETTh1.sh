#!/usr/bin/env bash
# ============================================================================
# ETTh1 调参（第二阶段，针对系统性偏差项 pred_len 336/720）
# 与默认参数(run_ETTh1.sh, --des Exp)的区别:
#   --train_epochs / --patience / --lradj 三项（见下方"调参旋钮"），并用 --des 隔离输出。
# 根因假设: type1 lr 每 epoch 减半 + epochs=10 + patience=3 → 长 horizon 欠拟合。
# 用法:
#   conda activate iTransformer
#   bash scripts/reproduce/tune_ETTh1.sh            # 跑 336 720
#   bash scripts/reproduce/tune_ETTh1.sh 336        # 只跑 336
#   GPU=1 bash scripts/reproduce/tune_ETTh1.sh      # 用 1 号卡
#   开发机自测: PY="D:/SoftWare22/miniconda/envs/itransformer/python.exe" bash scripts/reproduce/tune_ETTh1.sh 336
# 消融: 只改下方 4 个旋钮(如 LRADJ=type1 DES=TuneE20p5)即跑单变量对照, 不动循环体。
# 输出: checkpoints/{...TuneC20p5...}/ + result_long_term_forecast.txt(追加, des=TuneC20p5) + logs/
# 前置: 先 cp result_long_term_forecast.txt result_long_term_forecast.default.txt 备份默认结果。
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

PLS="${*:-336 720}"
mkdir -p logs
echo "============================================================"
echo " ETTh1 调参 | pred_len=[$PLS] | ep=$EPOCHS pa=$PATIENCE lradj=$LRADJ des=$DES | GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for pl in $PLS; do
  # ETTh1: pred_len 96/192 用 d_model=256; 336/720 用 d_model=512
  if (( pl <= 192 )); then DM=256; else DM=512; fi
  echo "---- ETTh1 pred_len=$pl d_model=$DM ----"
  "$PY" -u run.py --is_training 1 \
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
    --train_epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --lradj "$LRADJ" \
    --des "$DES" \
    --itr 1 \
    2>&1 | tee logs/tune_ETTh1_${DES}_pl${pl}.log
done

echo "============================================================"
echo " ETTh1 调参完成 | des=$DES | 结果见 result_long_term_forecast.txt 与 logs/"
echo "============================================================"
