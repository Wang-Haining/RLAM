#!/bin/sh
#SBATCH --job-name=rl_am_gemma-2b_w_sd_reward
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

JOB_NAME="rl_am_gemma-2b_w_sd_reward"
mkdir -p logs
mkdir -p ckpts

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

accelerate launch --main_process_port 29502 --config_file runs/ds_config.yaml \
rlam.py \
--job_name $JOB_NAME \
--print_sample_output_freq 0 \
--deepspeed \
--offload \
--lr 1e-6 \
--warm_up_steps 50 \
--world_size 2 \
--num_train_epochs 5 \
--base_model 'google/gemma-2b' \
--response_length 241 \
--truncate_token eos \
--truncate_token_id 1 \
--temperature 0.5 \
--sft_model_path "ckpts/sft_gemma-2b/checkpoint-1680" \
--logging_steps 2 \
--save_steps 20 \
--eval_steps 20 \
--num_eval_samples 64 \
--save_total_limit 30 \
--local_rollout_forward_batch_size 16 \
--gradient_accumulation_steps 4 \
--local_micro_batch_size 4 \
--local_eval_batch_size 4 \
--rlam.target_kl 2.0 \
--rlam.kl_coef 0.8 \
--rlam.kl_coef_lower_bound 0.75 \
--rlam.kl_coef_upper_bound 0.85 \
--rlam.k_beta 1e-4 \
--rlam.sl_coef 1.0 \
--rlam.wa_coef 2.5 \
--rlam.sd_coef 2.0 \
--rlam.swa_std_coef 0.0
