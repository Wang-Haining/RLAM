#!/bin/sh
#SBATCH --job-name=main_ppo_uam_olmo_waCoef2.1_target5_lr1e-6_klCoef0.19_v2
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
--save_folder main_ppo_uam_olmo_waCoef2.1_target5_lr1e-6_klCoef0.19_v2 \
--sft_ckpt_path ckpts/sft_OLMo-1B-hf/checkpoint-1100 \
--early_stopping true \
--target 5.0 \
--reward uam \
--wa_coef 2.1 \
--learning_rate 1e-6 \
--init_kl_coef 0.19 \
--num_saved_ckpts 30