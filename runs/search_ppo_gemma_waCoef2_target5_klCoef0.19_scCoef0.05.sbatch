#!/bin/sh
#SBATCH --job-name=search_ppo_gemma_waCoef2_target5_klCoef0.19_scCoef0.05
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
python -m ppo \
--save_folder search_ppo_gemma_waCoef2_target5_klCoef0.19_scCoef0.05 \
--sft_ckpt_path ckpts/sft_gemma-2b/checkpoint-1120 \
--early_stopping true \
--target 5.0 \
--reward uam \
--wa_coef 2.0 \
--sc_coef 0.05 \
--learning_rate 5e-6 \
--init_kl_coef 0.19 \
--num_saved_ckpts 30