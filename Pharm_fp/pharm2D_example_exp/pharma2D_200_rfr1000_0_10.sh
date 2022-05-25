#!/bin/bash

#SBATCH -p skylake-himem
#SBATCH --nodes=1
#SBATCH --job-name=0_10   ## Name of the job
#SBATCH --output=pharma2D_200_rfr1000_0_10.txt    ## Output file
#SBATCH --time=12:00:00           ## Job Duration
#SBATCH --ntasks=1             ## Number of tasks (analyses) to run
#SBATCH --mem=240000


## Execute the python script and pass the argument/input '90'
srun python3 pharma2D_200_rfr1000_0_10.py