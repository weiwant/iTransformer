# CLAUDE.md — iTransformer 复现项目指导

> 本文件是本项目的"项目记忆"与操作规范，供 Claude（开发机一侧）和协作者参考。每次会话自动加载。

> 📌 **当前阶段（新会话先看）**：**全表复现完成并结案**（长预测 36/36 + PEMS 16/16）。长预测偏差全部 <8%（33/36 <3%）。**PEMS pl48/96 发散已修复**：根因=训练无梯度裁剪 + 官方高 lr（epoch1→2 梯度爆炸）；修复（`--grad_clip 1.0` + lr 减半，commit `5624d85`）后发散全消除，PEMS 16 项 11✅/1🔶/4⚠️；残余 4 项（PEMS03/08 的 pl48/96, +30~57%）经增大 epoch + 多 seed 双重验证为配置极限（非 bug）。详见 §8.10 与 `实验记录.md` §7.3。

## 1. 项目概述

- **目标**：复现论文 *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*（ICLR 2024 Spotlight）的 **Table 1 主表**：多变量预测（features=M, seq_len=96），指标 MSE / MAE。
  - **长预测**（pred_len 96/192/336/720）：ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity(ECL)/Traffic，**+ Exchange Rate / Solar-Energy**（共 9 个）。
  - **短预测**（pred_len 12/24/48/96）：PEMS（4 子集 03/04/07/08，参数各异）。
- **源码**：来自官方仓库 thuml/iTransformer（基于 Time-Series-Library 框架）。
- **进度台账**：见同目录 `实验记录.md`（唯一的实验结果记录，持续更新）。
- **参照记录**：早期复现（2026-05-15，RTX 2060）记录在 `D:\wqWork\论文\开题报告\周报\0520\T4_iTransformer复现\`（实验记录.md / 实验操作手册.md / 图表解读.md）。

## 2. 工作流：开发机 ↔ GitHub ↔ 服务器

| 角色 | 硬件 | 职责 |
|---|---|---|
| **开发机（本机）** | RTX 2060 6GB | 写/改代码与脚本、修 bug、小数据集（ETTh1 等）冒烟测试、git 版本控制与推送 |
| **GitHub** | — | 代码中转：开发机 push → 服务器 pull |
| **服务器** | 2× RTX 4090（每卡 24GB） | 搭环境、跑全 7×4 表实验（ECL 321 / Traffic 862 等大实验） |

**结果流转约定**：数据集、checkpoint、结果（results/test_results）**不进 git**（见 §9）。服务器跑出的指标（MSE/MAE）**手动记录到 `实验记录.md`**；若需回传 pred/true，单独拷贝。

## 3. 环境

### 开发机（已就绪）
- conda env：`itransformer`（Python 3.10），安装位置 `D:\SoftWare22\miniconda\envs\itransformer`。
- **运行 python（重要，避免每次试错）**：本机 Git Bash / Claude Code 的 bash 工具里 **`conda` 不在 PATH**，`conda activate`、`conda run -n itransformer` 都会报 `command not found`。**直接用 env 的解释器绝对路径调用**（首选方式）：
  ```bash
  "D:/SoftWare22/miniconda/envs/itransformer/python.exe" run.py --is_training 1 ...
  "D:/SoftWare22/miniconda/envs/itransformer/python.exe" scripts/reproduce/extract_results_full.py
  ```
  > 该 env 是本机唯一带 cu118 torch + 全部依赖的解释器；用裸 `python` 可能落到别的默认/系统环境而出错。本机所有 conda env 列表见 `~/.conda/environments.txt`（miniconda 装在 `D:\SoftWare22\miniconda`）。
- torch 2.0.0+cu118；numpy 1.23.5, pandas 1.5.3, scikit-learn 1.2.2, matplotlib 3.7.0, reformer-pytorch 1.4.4
- torch 安装：cu118 版**单独装**（`requirements.txt` 已移除 torch，避免覆盖 GPU 版）：
  ```bash
  pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
  ```
  > ⚠️ 历史：旧 `requirements.txt` 含 `torch==2.0.0`，`pip install -r requirements.txt` 会覆盖 cu118 torch，导致 `import torch` 报 `undefined symbol: iJIT_NotifyEvent`（conda/pip 混装 mkl 冲突）。已修复。

### 服务器（已就绪：2× RTX 4090，Ubuntu）
- conda env `iTransformer`（Python 3.10）+ pip cu118 torch；其余依赖 `pip install -r requirements.txt`（已去 torch）。
- **LD_LIBRARY_PATH 已永久化**：`scripts/reproduce/env_setup.sh` 在每个脚本开头自动设置（解决 pip torch 的 cuDNN 找不到 `libnvrtc.so`）；无需手动 export。
- 跑实验：`conda activate iTransformer && bash scripts/reproduce/run_<dataset>.sh`（详见 `scripts/reproduce/README.md`）。
- 多 GPU：`--use_multi_gpu --devices 0,1`（`nn.DataParallel`）；但复现优先保持官方 batch_size，单卡即可，多卡主要用于并行跑不同数据集（`GPU=0 ...` / `GPU=1 ...`）。

## 4. 快速命令

入口统一是 `python -u run.py`（Windows 用 `set CUDA_VISIBLE_DEVICES=0`，Linux/bash 用 `export`）。下文示例为命令骨架；**在本机 bash 里实际执行时，把 `python` 换成 §3 的 env 绝对路径** `"D:/SoftWare22/miniconda/envs/itransformer/python.exe"`（`conda` 不在 PATH）。服务器上则先 `conda activate iTransformer` 再用 `python`。

**训练 + 自动测试**（`--is_training 1`）：
```bash
python -u run.py --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96 --model iTransformer --data ETTh1 \
  --features M --seq_len 96 --pred_len 96 \
  --e_layers 2 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 256 --d_ff 256 --des Exp --itr 1
```
产出：`./checkpoints/{setting}/checkpoint.pth`、`./results/{setting}/{metrics,pred,true}.npy`、`./test_results/{setting}/*.pdf`，并追加一行到 `result_long_term_forecast.txt`。

**仅加载 checkpoint 测试**（`--is_training 0`，用于中断续测或单独评估）：
```bash
python -u run.py --is_training 0 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id ETTh1_96_96 --model iTransformer --data ETTh1 \
  --features M --seq_len 96 --pred_len 96 \
  --e_layers 2 --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 256 --d_ff 256 --des Exp --itr 1
```
从 `./checkpoints/{setting}/checkpoint.pth` 加载。**参数必须与训练时完全一致**，否则 `setting` 对不上、找不到权重。

批量复现直接用脚本：`bash ./scripts/multivariate_forecasting/<Dataset>/iTransformer*.sh`。

## 5. 代码结构

- `run.py` — 入口，argparse 参数解析；按 `--is_training` 走训练或仅测试分支。
- `experiments/exp_long_term_forecasting.py` — `Exp_Long_Term_Forecast`：`train()`（含 vali 早停）、`test()`、`vali()`、`predict()`。
- `experiments/exp_long_term_forecasting_partial.py` — 部分变量训练（`--exp_name partial_train`）。
- `model/iTransformer.py` — 核心：实例归一化 → 倒置 embedding（变量作 token）→ encoder → 线性 projector → 反归一化。
- `data_provider/` — `data_factory.py`（按 `--data` 选数据集类）、`data_loader.py`（ETT/Custom/Solar/PEMS/Pred）。test 时 batch_size=1、drop_last=True。
- `layers/` — Embedding（`DataEmbedding_inverted`）、Transformer encoder、SelfAttention。
- `utils/` — EarlyStopping、adjust_learning_rate、visual、metrics。
- `scripts/` — 各实验脚本（`multivariate_forecasting/` 为主表）。
- `scripts/reproduce/` — **本项目复现配套脚本**（见 `scripts/reproduce/README.md`）：`env_setup.sh`（公共环境，自动设 LD_LIBRARY_PATH）、`smoke_test.sh`（冒烟测试）、`run_<dataset>.sh`（按数据集训练）。**长预测 9 个**：ETTh1/ETTh2/ETTm1/ETTm2/Weather/ECL/Traffic/Exchange/Solar；**短预测 1 个**：`run_PEMS.sh`（首参为子集号，参数经 `pems_params` 查表，勿手填）。

## 6. 复现策略与约定

1. **第一阶段（严格复现）**：用官方默认参数（见 §7）跑全 7×4 表，如实记录每个 MSE/MAE 及与论文的偏差。**已完成**（长预测 36/36 + PEMS 16/16）。
2. **第二阶段（调参修正）→ 已结案**：原计划对长预测偏差项调参，但 2026-06-30 复盘查明"偏差"源自论文参考值误抄（非欠拟合），订正后全部 <8%，**调参无必要**（详见 `实验记录.md` §7.2）。调参脚本 `tune_*.sh` / cosine 分支保留备用。
3. 复现成功判定：MSE 偏差 |·|<8% 达标、<3% 优秀。合理基准是论文自身种子波动（Table 5 用 5 种子平均）+ 硬件浮点差异，刻意不为 2~3% 微差调参。
4. 所有新结果写入 `实验记录.md` 的进度表，并在"变更日志"加一行。

## 7. 数据集与超参对照表（取自官方脚本，务必照此填参数）

| 数据集 | --data | --data_path | enc_in | e_layers | d_model | d_ff | batch_size | lr |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | ETTh1 | ETTh1.csv | 7 | 2 | 256(96/192)·512(336/720) | 同 d_model | 32 | 0.0001 |
| ETTh2 | ETTh2 | ETTh2.csv | 7 | 2 | 128 | 128 | 32 | 0.0001 |
| ETTm1 | ETTm1 | ETTm1.csv | 7 | 2 | 128 | 128 | 32 | 0.0001 |
| ETTm2 | ETTm2 | ETTm2.csv | 7 | 2 | 128 | 128 | 32 | 0.0001 |
| Weather | custom | weather.csv | 21 | 3 | 512 | 512 | 32 | 0.0001 |
| Electricity(ECL) | custom | electricity.csv | 321 | 3 | 512 | 512 | 16 | 0.0005 |
| Traffic | custom | traffic.csv | 862 | 4 | 512 | 512 | 16 | 0.001 |
| Exchange | custom | exchange_rate.csv | 8 | 2 | 128 | 128 | 32 | 0.0001 |
| Solar | Solar | solar_AL.txt | 137 | 2 | 512 | 512 | 32 | 0.0005 |

ETT 类 root_path=`./dataset/ETT-small/`；其余 root_path=`./dataset/<name>/`（Exchange→`exchange_rate/`、Solar→`Solar/`首字母大写、PEMS→`PEMS/`）。

> **PEMS（短预测，pred_len 12/24/48/96，非长预测）**：4 子集参数各异，详见 `scripts/reproduce/run_PEMS.sh` 内 `pems_params` 查表（enc_in：03=358 / 04=307 / 07=883 / 08=170；e_layers / batch / lr / use_norm 随子集与 pred_len 变）。直接用脚本，勿手填。

通用默认（脚本未显式指定时）：`seq_len=96 label_len=48 train_epochs=10 patience=3 lradj=type1 n_heads=8 dropout=0.1 embed=timeF use_norm=True factor=1 activation=gelu`。

> 注意：**只有 ETTh1 用 d_model 256/512**，ETTh2/ETTm1/ETTm2 一律 128。填错会导致 `setting` 目录名不同、结果无法对照。

## 8. 已知坑与注意事项

1. **CPU torch**：requirements.txt 装 CPU 版，须手动换 cu118（见 §3）。
2. **setting 目录名"前缀错位"**：`run.py` 拼出的目录名形如 `ETTh1_96_96_iTransformer_ETTh1_M_ft96_sl48_ll96_pl256_dm8_nh2_el1_dl256_...`，其前缀（ft/sl/ll/pl/dm/nh/el…）与所标参数**错位**（`sl48`=label_len、`pl256`=d_model、`dm8`=n_heads…）。**不要按前缀字面理解**，照训练脚本原样填参数即可；仅测试时参数必须与训练逐字一致。
3. **~~长预测偏差~~（已查明）**：曾见 ETTh1 336/720 "偏差 +13%/+18%"，疑欠拟合；实为**论文参考值误抄**（脚本 PAPER 写成 0.432/0.431，真实 0.487/0.503）。已订正（Table 10，commit `61ba882`），订正后偏差 +0.8%/+1.3%。调参实验证实 best epoch=1、无调参空间。见 `实验记录.md` §7.2。
4. **数据集不入 git**：服务器需自行下载（百度网盘 pwd `9qjr` 或论文 README 的 Google Drive），解压到 `./dataset/`。
5. **中断续测**：训练被意外关闭（如 Traffic）但已存 checkpoint，用 `--is_training 0` 直接加载评估。
6. **result_long_term_forecast.txt 是追加写入**：会累积历史结果，区分新旧需看 setting 名。
7. **脚本里 CUDA_VISIBLE_DEVICES 可能写死成 1/2**：本机单卡改成 0。
8. **PEMS 是短预测**：pred_len=12/24/48/96（非主表长预测 96/192/336/720），4 子集参数各异，用 `run_PEMS.sh <子集号>`；勿套用长预测参数。
9. **Exchange train_epochs 笔误**：官方 `Exchange/iTransformer.sh` 的 pred_len=336 段含 `--train_epochs 1`（疑笔误），`run_Exchange.sh` 已修正为默认 10。
10. **PEMS pl48/96 训练发散（已修复，2026-06-30）**：iTransformer 训练循环原本**无梯度裁剪**（`exp_long_term_forecasting.py` 的 `loss.backward(); model_optim.step()`），PEMS 在官方高 lr(0.001) + 长 pred_len(`projector=Linear(d_model,pred_len)` 输出维度大) 下，**epoch1→2 之间单步梯度爆炸**，vali loss 翻倍飙升、best 永远卡 epoch1，导致 PEMS03 pl96 +918% / PEMS07 +698% / PEMS08 +99%。根因**非硬件**（本机 RTX2060 完整复现）。修复（已入 `run_PEMS.sh` 查表）：pl48/96 加 `--grad_clip 1.0` + lr 减半到 0.0005（PEMS04 本就 0.0005，只加 clip）；pl12/24 保持原参不动（已达标）。`run.py` 新增 `--grad_clip`（默认 0=关闭，**不影响长预测已复现结果**）。残余 4 项（PEMS03/08 的 pl48/96）经增大 epoch（`EPOCHS=30`，best epoch 11 早停）+ 多 seed（`ITR=5`，5 seed 均值 0.258、方差仅 ±0.003）双重验证，确认是该配置（d_model=512 / lr=0.0005）的单 seed 客观极限，**非 bug、非欠拟合**；论文靠 5 seed 平均 + 可能有未公开细节，刻意不为此偏离官方参数调参。`run.py` 多 seed（seed=2023+ii + cuda seed，commit `57cc6c9`）、`run_PEMS.sh` 的 `EPOCHS`/`ITR` 环境变量（commit `88e4d6a`）保留备用。复盘见 `实验记录.md` §7.3。

## 9. Git 与版本控制

- `origin` = `git@github.com:weiwant/iTransformer.git`（SSH，已配置可用；本机无 HTTPS credential helper，统一用 SSH）。
- **复现工作统一在 `reproduce` 分支进行**（已从 main 创建并推送，本地 `reproduce` 跟踪 `origin/reproduce`）。
- 日常提交流程：
  ```bash
  git checkout reproduce
  git add <代码/脚本/文档>
  git commit -m "<message>"
  git push
  ```
- **提交范围**：仅代码、scripts/、文档（CLAUDE.md、实验记录.md）、.gitignore。
- **不提交**：dataset/、checkpoints/、results/、test_results/、result_long_term_forecast.txt、*.pth、*.npy（见 .gitignore）。
- 提交前务必 `git status` 复核，避免误提交大文件。

## 10. 文档索引

- `实验记录.md`（本项目根目录）— 实验进度台账，唯一的结果记录处。
- `D:\wqWork\论文\开题报告\周报\0520\T4_iTransformer复现\` — 早期复现三份文档（实验记录/操作手册/图表解读）。
- 论文：https://arxiv.org/abs/2310.06625
- 上游仓库：https://github.com/thuml/iTransformer
