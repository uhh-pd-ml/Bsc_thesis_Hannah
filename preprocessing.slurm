#!/bin/bash
#SBATCH --job-name=prepreprocessing
#SBATCH --output=/beegfs/u/bbd1146/prepreprocessing_output%J.log
#SBATCH --error=/beegfs/u/bbd1146/prepreprocessing_error%J.log
#SBATCH --time=12:00:00
#SBATCH --partition=std
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=all
#SBATCH --mail-user=hannah.wittboldt-mueller@studium.uni-hamburg.de




module load python
source /usw/u/bbd1146/my_venv/bin/activate
python prepreprossing.py
