#!/bin/sh
#SBATCH --job-name=ppo_flant5_wa_coef2_target5_lr5e-6_sstTrue
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
python -m ppo --save_folder ppo_flant5_wd_coef2_target5_lr5e-6_sstTrue \
--sft_ckpt_path ckpts/sft_flan-t5-xl/checkpoint-3940 \
--early_stopping true \
--wa_coef 2.0 \
--target 5.0 \
--learning_rate 5e-6 \
--skip_special_tokens true