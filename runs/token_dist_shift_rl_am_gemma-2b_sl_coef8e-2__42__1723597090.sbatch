#!/bin/sh
#SBATCH --job-name=token_dist_shift_rl_am_gemma-2b_sl_coef8e-2__42__1723597090
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0

. .venv/bin/activate
python -m token_dist_shift \
--csv_file "eval_results_temp_0.01/ckpts|rl_am_gemma-2b_sl_coef8e-2__42__1723597090|step_1250_ari_13.64.csv" \
--output_json "eval_results_temp_0.01/tds_ckpts|rl_am_gemma-2b_sl_coef8e-2__42__1723597090|step_1250_ari_13.64.json"
