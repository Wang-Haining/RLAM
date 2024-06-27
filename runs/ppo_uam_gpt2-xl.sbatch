#!/bin/sh
#SBATCH --job-name=ppo_uam_gpt2-xl
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

JOB_NAME="ppo_uam_gpt2-xl"
mkdir -p logs
mkdir -p ckpts

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

accelerate launch --config_file runs/ds_config.yaml \
ppo.py \
--job_name $JOB_NAME \
--print_sample_output_freq 0 \
--deepspeed \
--offload \
--lr 1e-4 \
--warm_up_steps 50 \
--world_size 2 \
--num_train_epochs 5 \
--base_model 'openai-community/gpt2-xl' \
--response_length 244 \
--truncate_token eos \
--truncate_token_id 50256 \
--temperature 0.7 \
--penalty_reward_value -5 \
--non_eos_penalty \
--sft_model_path "ckpts/sft_gpt2-xl/checkpoint-1880" \
--logging_steps 2 \
--save_steps 20 \
--eval_steps 20 \
--num_eval_samples 64 \
--save_total_limit 30 \
--local_rollout_forward_batch_size 16 \
--gradient_accumulation_steps 4 \
--local_micro_batch_size 2 \
--local_eval_batch_size 1 \
--rluam.sl_coef 1.0 \
--rluam.wa_coef 2.05
