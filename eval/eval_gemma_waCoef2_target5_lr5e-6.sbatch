#!/bin/sh
#SBATCH --job-name=eval_gemma_waCoef2_target5_lr5e-6
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-4:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
python -m eval_outputs --ckpt_path ckpts/ppo_gemma_wa_coef2_target5_lr5e-6/model_epoch_4_step_60_ari_13.68.pt