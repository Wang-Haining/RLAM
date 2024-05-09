#!/bin/sh
#SBATCH --job-name=ppo_gemma_word_difficulty_reward_coef0.5
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
python -m ppo --save_folder ppo_gemma_word_difficulty_reward_coef0.5 --sft_ckpt_path ckpts/sft_gemma-2b/checkpoint-1120 --early_stopping true --wd_coef 0.5