#!/bin/sh
#SBATCH --job-name=sft_gemma-7b
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
accelerate launch --main_process_port 29501 --config_file runs/ds_config.yaml \
sft_ds.py --model gemma-7b --per_device_train_batch_size 1 --learning_rate 3e-6 --gradient_checkpointing --deepspeed
