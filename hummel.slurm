#!/bin/bash
#SBATCH --job-name=hummel_run
#SBATCH --output=/beegfs/u/bbd1146/hummel_output%J.log
#SBATCH --error=/beegfs/u/bbd1146/hummel_error%J.log
#SBATCH --time=4:40:00
#SBATCH --partition=std
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=all
#SBATCH --mail-user=hannah.wittboldt-mueller@studium.uni-hamburg.de




module load python
source /beegfs/u/bbd1146/temp_install/bin/activate
python ae_prepro.py
