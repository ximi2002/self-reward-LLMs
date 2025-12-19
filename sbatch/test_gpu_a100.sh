#!/bin/bash
#SBATCH --job-name=test_gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --account=heng-prj-aac
#SBATCH --output=test_gpu_a100_%j.out
#SBATCH --error=test_gpu_a100_%j.err

set -euo pipefail

hostname
nvidia-smi