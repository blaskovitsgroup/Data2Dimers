#!/bin/bash

# example submission script

#SBATCH --job-name=x
#SBATCH --account=x
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=8       
#SBATCH --mem=16G                   
#SBATCH --time=0-06:00              
#SBATCH --output=chemberta_%j.out    
#SBATCH --error=chemberta_%j.err  

module load python/3.11.5
module load scipy-stack
module load rdkit
module load arrow/19.0.1

source PATH_TO/chemberta_env/bin/activate

python pretrain.py
