#!/bin/sh
#SBATCH --job-name=ppo_gemma-2b
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

accelerate launch --config_file runs/ds_config.yaml \
ppo.py \
--project_name RLUAM \
--run_name ppo_gemma-2b \
--print_sample_output_freq 0 \
--deepspeed \
--offload \
--lr 5e-7 \
--warm_up_steps 50 \
--world_size 2 \
--num_train_epochs 5 \
--base_model 'google/gemma-2b' \
--response_length 241 \
--truncate_token eos \
--truncate_token_id 1 \
--temperature 0.7 \
--penalty_reward_value -5 \
--non_eos_penalty \
--sft_model_path "ckpts/sft_gemma-2b/checkpoint-1120" \
--logging_steps 2 \
--save_steps 10 \
--eval_steps 10 \
--num_eval_samples 64 \
--save_total_limit 30 \
--output_dir 'ckpts/ppo_gemma-2b' \
--local_rollout_forward_batch_size 16 \
--gradient_accumulation_steps 4 \
--local_micro_batch_size 2 \
--local_eval_batch_size 1 \
--rluam..target_kl 4.0 \
--rluam..k_beta 0.1 \
--rluam.sl_coef 1.0 \
--rluam.wa_coef 2.05

