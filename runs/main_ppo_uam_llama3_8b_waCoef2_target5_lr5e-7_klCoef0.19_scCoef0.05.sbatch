#!/bin/sh
#SBATCH --job-name=main_ppo_uam_llama3_8b_waCoef2_target5_lr5e-7_klCoef0.19_scCoef0.05
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
--save_folder main_ppo_uam_llama3_8b_waCoef2_target5_lr5e-7_klCoef0.19_scCoef0.05 \
--sft_ckpt_path ckpts/sft_Meta-Llama-3-8B/checkpoint-1840 \
--early_stopping true \
--target 5.0 \
--reward uam \
--wa_coef 2.0 \
--sc_coef 0 \
--learning_rate 5e-7 \
--init_kl_coef 0.19 \
--is_peft_model true \
--num_saved_ckpts 30