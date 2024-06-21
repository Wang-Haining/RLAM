#!/bin/sh
#SBATCH --job-name=ppo_olmo-1b
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

accelerate launch --config_file runs/deepspeed.yaml \
ppo.py \
--project_name RLUAM \
--run_name ppo_olmo-1b \
--print_sample_output_freq 0 \
--deepspeed \
--offload \
--lr 3e-6 \
--warm_up_steps 50 \
--world_size 2 \
--num_train_epochs 5 \
--base_model 'allenai/OLMo-1B-hf' \
--response_length 223 \
--truncate_token eos \
--truncate_token_id 50279 \
--temperature 0.7 \
--penalty_reward_value -5 \
--non_eos_penalty \
--sft_model_path "ckpts/sft_OLMo-1B-hf/checkpoint-1100" \
--sl_coef 1.0 \
--wa_coef 2.0 \
--logging_steps 2 \
--save_steps 10 \
--eval_steps 10 \
--num_eval_samples 64 \
--save_total_limit 3 \
--output_dir 'ckpts/ppo_olmo-1b' \
--local_rollout_forward_batch_size 64 \
--gradient_accumulation_steps 16 \
--local_micro_batch_size 2 \
--local_eval_batch_size 1
