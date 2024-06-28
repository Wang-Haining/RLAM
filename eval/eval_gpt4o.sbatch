#!/bin/sh
#SBATCH --job-name=eval_gpt4o
##SBATCH --account=group-jasonclark
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0-4:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0

. .venv/bin/activate
python -m eval_gpt4o
