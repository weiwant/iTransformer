#!/usr/bin/env bash
# ============================================================================
# scripts/reproduce/env_setup.sh — 复现脚本公共环境初始化
# ----------------------------------------------------------------------------
# 作用：在 Linux 上把 pip torch 自带的 CUDA 库目录加入 LD_LIBRARY_PATH，
#       解决 cuDNN 加载时报 "libnvrtc.so: cannot open shared object file" 的问题
#       （pip 的 nvidia-* 库装在 site-packages/nvidia/*/lib/，默认搜不到）。
#       Windows / macOS 不用此机制，自动跳过（无副作用）。
#
# 用法：每个复现脚本开头一行 ——
#   source "$(dirname "${BASH_SOURCE[0]}")/env_setup.sh"
#
# 前提：已 conda activate iTransformer（本脚本依赖 python 能 import nvidia）。
# ============================================================================

if [[ "$(uname -s)" == "Linux" ]]; then
  # 定位 pip torch 的 nvidia 库目录；失败则跳过（不中断脚本）
  NVLIB="$(python -c "import nvidia,glob,os;print(':'.join(glob.glob(os.path.join(os.path.dirname(nvidia.__file__),'*/lib'))))" 2>/dev/null || true)"
  if [[ -n "$NVLIB" ]]; then
    export LD_LIBRARY_PATH="$NVLIB:${LD_LIBRARY_PATH:-}"
    echo "[env_setup] 已将 pip torch 的 CUDA 库加入 LD_LIBRARY_PATH"
  else
    echo "[env_setup] 未检测到 pip nvidia 库（可能用 conda 装的 torch），跳过 LD_LIBRARY_PATH 设置"
  fi
fi
