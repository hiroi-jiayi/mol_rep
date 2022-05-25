#!/bin/bash

#SBATCH -p cclake-himem
#SBATCH --job-name=n_rdk2_1   ## Name of the job
#SBATCH --output=rdkit2_200_rfr1000_1.txt    ## Output file
#SBATCH --time=12:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --nodes=1


## Execute the python script and pass the argument/input '90'
srun python3 rdkit2_200_rfr1000_1.py