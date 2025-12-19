#!/bin/bash

#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=high
#SBATCH --time=8:00:00
#SBATCH --chdir=/nfshomes/suzhang1/PhD/848K             # CBCB 文档明确要求用 cbcb 账号           # TODO: 如果集群要求账号/项目号
#SBATCH --output=logs/sft_dpo_%j.out
#SBATCH --error=logs/sft_dpo_%j.err

set -euo pipefail

# 1) 模块/环境
module purge
if command -v module >/dev/null 2>&1; then
  CUDA_MODULE="${CUDA_MODULE:-cuda/12.1.1}"
  if module -t avail "$CUDA_MODULE" 2>/dev/null | grep -q "$CUDA_MODULE"; then
    module load "$CUDA_MODULE"
  else
    for m in cuda/12.4 cuda/12.2 cuda/12.1 cuda/11.8; do
      if module -t avail "$m" 2>/dev/null | grep -q "^$m$"; then
        module load "$m"
        break
      fi
    done
  fi
fi

#################### 2) Conda 环境 ####################
# 把这里改成你新装的 Miniconda 路径，比如 /fs/nexus-scratch/$USER/conda/miniconda3
CONDA_ROOT="${CONDA_ROOT:-/fs/nexus-scratch/$USER/miniconda3}"
CONDA_ENV="${CONDA_ENV:-848k}"

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  echo "Missing conda.sh at $CONDA_ROOT; set CONDA_ROOT/CONDA_ENV accordingly." >&2
  exit 1
fi

# 2) 可选：HF 私有/受限模型 token
# export HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 3) 进入代码目录
cd ~/PhD/848K  # TODO: 改成你的项目路径

# 4) 运行一键多轮（首轮 SFT + 2 轮自奖励+DPO），demo 模式
srun python scripts/full_iterate_self_reward.py \
  --rounds 2 \
  --stage dpo \
  --model-path Qwen/Qwen2.5-VL-7B-Instruct \
  --precision bf16 \
  --mode normal
