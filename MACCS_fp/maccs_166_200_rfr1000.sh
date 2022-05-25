#!/bin/bash

#SBATCH -p skylake-himem
#SBATCH --nodes=1
#SBATCH --job-name=maccs_166_200_rfr1000   ## Name of the job
#SBATCH --output=maccs_166_200_rfr1000.txt    ## Output file
#SBATCH --time=12:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run


## Execute the python script and pass the argument/input '90'
srun python3 maccs_166_200_rfr1000.py