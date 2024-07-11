#!/bin/sh
#SBATCH --job-name=ppo_uam_olmo-1b_dynamic_kl_control
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

JOB_NAME="ppo_uam_olmo-1b_dynamic_kl_control"
mkdir -p logs
mkdir -p ckpts

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate

accelerate launch --main_process_port 29501 --config_file runs/ds_config.yaml \
ppo.py \
--job_name $JOB_NAME \
--print_sample_output_freq 0 \
--deepspeed \
--offload \
--lr 5e-7 \
--warm_up_steps 50 \
--world_size 2 \
--num_train_epochs 5 \
--base_model 'allenai/OLMo-1B-hf' \
--response_length 223 \
--truncate_token eos \
--truncate_token_id 50279 \
--temperature 0.7 \
--sft_model_path "ckpts/sft_OLMo-1B-hf/checkpoint-1100" \
--logging_steps 2 \
--save_steps 20 \
--eval_steps 20 \
--num_eval_samples 64 \
--save_total_limit 30 \
--local_rollout_forward_batch_size 16 \
--gradient_accumulation_steps 4 \
--local_micro_batch_size 8 \
--local_eval_batch_size 8 \
--rluam.target_kl 2.0 \
--rluam.kl_coef 0.8 \
--rluam.kl_coef_lower_bound 0.75 \
--rluam.kl_coef_upper_bound 0.90 \
--rluam.k_beta 1e-3 \
--rluam.sl_coef 1.0 \
--rluam.wa_coef 2.1

