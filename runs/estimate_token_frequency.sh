#!/bin/sh
#SBATCH --job-name=estimate_wiki_word_frequency
##SBATCH --account=group-jasonclark
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0

. .venv/bin/activate
python -m estimate_wiki_word_frequency