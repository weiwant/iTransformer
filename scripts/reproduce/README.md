# scripts/reproduce/ — iTransformer 复现配套脚本

本目录存放本项目（复现 iTransformer Table 1）的配套脚本，与上游官方脚本 `scripts/multivariate_forecasting/` 区分。

## 脚本一览

| 脚本 | 用途 |
|---|---|
| `env_setup.sh` | **公共环境初始化**（被下面所有脚本自动 `source`）：Linux 下把 pip torch 的 CUDA 库加入 `LD_LIBRARY_PATH`，解决 cuDNN 找不到 `libnvrtc.so`。一般不单独跑。 |
| `smoke_test.sh` | 冒烟测试：ETTh1 + pred_len=96 + train_epochs=1 跑通全流程，验证环境。 |
| `run_ETTh1.sh` | ETTh1 训练（pred_len 96/192/336/720） |
| `run_ETTh2.sh` | ETTh2 训练 |
| `run_ETTm1.sh` | ETTm1 训练 |
| `run_ETTm2.sh` | ETTm2 训练 |
| `run_Weather.sh` | Weather 训练（21 变量） |
| `run_ECL.sh` | Electricity 训练（321 变量） |
| `run_Traffic.sh` | Traffic 训练（862 变量，最慢） |

## 前置条件

1. **激活环境**：`conda activate iTransformer`（开发机环境名为 `itransformer`）。
2. **数据集**：已下载并解压到 `./dataset/`（见 CLAUDE.md §8）。数据集不进 git。

## 用法

```bash
conda activate iTransformer

# 冒烟测试（验证环境）
bash scripts/reproduce/smoke_test.sh

# 按数据集训练（每个脚本跑该数据集 4 个 pred_len）
bash scripts/reproduce/run_ETTh1.sh            # ETTh1: 96/192/336/720
bash scripts/reproduce/run_Traffic.sh 720      # 只跑 Traffic 720（久的单独跑）
GPU=1 bash scripts/reproduce/run_ECL.sh        # 用 1 号卡
```

每个 `run_*.sh` 都支持：
- 默认跑该数据集全部 4 个 pred_len（96/192/336/720）；
- 传参指定 pred_len：`bash run_XX.sh 96 192`；
- `GPU=N` 环境变量选卡（默认 0）。

## 关于 LD_LIBRARY_PATH（Linux 服务器）

pip 装的 torch 把 CUDA 库放在 `site-packages/nvidia/*/lib/`，Linux 默认搜不到，cuDNN 加载会报 `libnvrtc.so: cannot open shared object file`。`env_setup.sh` 会在每个脚本开头自动把这些目录加入 `LD_LIBRARY_PATH`（仅 Linux，Windows/macOS 跳过）。**无需手动 export**，直接跑脚本即可（看到 `[env_setup] 已将...` 提示即生效）。

## 结果位置

- `results/<setting>/metrics.npy` — `[mae, mse, rmse, mape, mspe]`（主用，MSE=`[1]`、MAE=`[0]`）。
- `results/<setting>/{pred,true}.npy` — 预测与真值。
- `checkpoints/<setting>/checkpoint.pth` — 模型权重。
- `result_long_term_forecast.txt` — 所有实验 MSE/MAE 汇总（追加写入）。

> 以上均在 `.gitignore`，不进 git。结果请手动回传并记入根目录 `实验记录.md`。

## 参数来源

各数据集超参取自官方 `scripts/multivariate_forecasting/<集>/iTransformer*.sh`（详见 CLAUDE.md §7）。通用默认 `train_epochs=10 patience=3 lradj=type1`（run.py 默认，脚本不显式传）。
