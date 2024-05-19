#!/bin/sh
#SBATCH --job-name=eval_eval_main_ppo_ari_gemma_target5_lr5e-6_klCoef0.19_v1
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
--ckpt_path ckpts/main_ppo_ari_gemma_target5_lr5e-6_klCoef0.19_v1/model_epoch_0_step_60_ari_16.89.pt