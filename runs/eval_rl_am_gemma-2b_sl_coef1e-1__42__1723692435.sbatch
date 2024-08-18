#!/bin/sh
#SBATCH --job-name=eval_rl_am_gemma-2b_sl_coef1e-1__42__1723692435
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate

python -m eval_outputs --upper_ari_bound 15.0  --lower_ari_bound 8.0 --batch_size 40 --temperature 0.7 --verbose --ckpt_path ckpts/rl_am_gemma-2b_sl_coef1e-1__42__1723692435 --sft_ckpt ckpts/sft_gemma-2b/checkpoint-1680/
