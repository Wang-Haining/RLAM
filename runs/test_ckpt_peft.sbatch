#!/bin/sh
#SBATCH --job-name=test_ckpt_peft_llama3-8b
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
python -m ppo \
--save_folder test_ckpt_peft_llama3-8b \
--sft_ckpt_path ckpts/sft_Meta-Llama-3-8B/checkpoint-1840 \
--early_stopping true \
--target 5.0 \
--reward uam \
--wa_coef 2.0 \
--sc_coef 0.0 \
--learning_rate 5e-6 \
--init_kl_coef 0.19 \
--num_saved_ckpts 4 \
--eval_interval 2 \
--num_eval_samples 2 \
--is_peft_model true