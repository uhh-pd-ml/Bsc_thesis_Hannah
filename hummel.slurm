#!/bin/bash
#SBATCH --job-name=hummel_run
#SBATCH --output=/beegfs/u/bbd1146/hummel_output%J.log
#SBATCH --error=/beegfs/u/bbd1146/hummel_error%J.log
#SBATCH --time=0:50:00
#SBATCH --partition=std
#SBATCH --cpus-per-task=16





module load python
source /beegfs/u/bbd1146/temp_install/bin/activate
#python roc_kurve_hardware/roc_kurve.py
#python Feature_plots/ae_feature_plots_sr2.py
#python ae_quhmain.py
python ae_shap2.py