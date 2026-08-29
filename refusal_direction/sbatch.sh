#!/bin/bash

#SBATCH --job-name=model_runs               # sets the job name
#SBATCH --output=model_runs.out             # indicates a file to redirect STDOUT to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=model_runs.out              # indicates a file to redirect STDERR to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --mail-type=ALL

# GAMMA Commands
#SBATCH --account=gamma
#SBATCH --partition=gamma 
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=30gb 
##SBATCH --nodelist=gammagpu[03]        # You can specify the example GPU node if you want

source ~/.bashrc

conda activate refusal
cd ~/steer-interp/refusal_direction

python -m pipeline.run_pipeline --model_path google/gemma-2b-it
# python -m pipeline.run_pipeline --model_path google/gemma-2-2b-it
# python -m pipeline.run_pipeline --model_path google/gemma-2-9b-it
# python -m pipeline.run_pipeline --model_path meta-llama/Llama-3.1-8B-Instruct