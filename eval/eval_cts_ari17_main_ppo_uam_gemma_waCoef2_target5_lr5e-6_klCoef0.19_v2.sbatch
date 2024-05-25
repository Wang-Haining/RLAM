#!/bin/sh
#SBATCH --job-name=eval_cts_ari17_main_ppo_uam_gemma_waCoef2_target5_lr5e-6_klCoef0.19_v2
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-1:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
python -m eval_outputs \
--ckpt_path ckpts/cts_ari17_main_ppo_uam_gemma_waCoef2_target5_lr5e-6_klCoef0.19_v2/model_epoch_0_step_80_ari_16.76.pt