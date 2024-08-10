#!/bin/sh
#SBATCH --job-name=rl_am_gemma-2b_w_wa_sl_reward
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

JOB_NAME="rl_am_gemma-2b_w_wa_sl_reward"
mkdir -p logs
mkdir -p ckpts

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

accelerate launch --main_process_port 29501 --config_file runs/ds_config.yaml \
rlam.py \
--job_name $JOB_NAME \
--print_sample_output_freq 0 \
--deepspeed \
--offload \
--lr 1e-6 \
--world_size 2 \
--base_model 'google/gemma-2b' \
--response_length 241 \
--truncate_token eos \
--truncate_token_id 1 \
--temperature 0.7 \
--sft_model_path "ckpts/sft_gemma-2b/checkpoint-1680" \
--logging_steps 2 \
--save_steps 50 \
--eval_steps 50 \
--num_eval_samples 64 \
--save_total_limit 30 \
--local_rollout_forward_batch_size 16 \
--gradient_accumulation_steps 4 \
--local_micro_batch_size 4 \
--local_eval_batch_size 4 \
--early_stop_patience 100 \
--rlam.kl_coef 0.2 \
--rlam.kl_coef_upper_bound 0.25 \
--rlam.kl_coef_lower_bound 0.15 \
--rlam.target_kl 8.0 \
--rlam.k_beta 1e-2 \
--rlam.sl_coef 5e-3 \
--rlam.wa_coef 4.0 \
--rlam.sd_coef 0.0 \
--rlam.swa_std_coef 0.0
