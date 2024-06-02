#!/bin/sh
#SBATCH --job-name=eval_main_ppo_gemma_2b_waCoef2_target5_klCoef0.19_scCoef0
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
--ckpt_path ckpts/main_ppo_gemma_2b_waCoef2_target5_klCoef0.19_scCoef0/model_epoch_9_step_0_ari_13.16.pt