#!/bin/bash

################################################################################################
### Generate reconstruction / swap samples used by AKD/AED evaluation.
### Submit as a Slurm job array; pass --export DATASET=<vox1|celebv|mug|taichi>.
################################################################################################

#SBATCH --partition main
#SBATCH --time 7-00:00:00
#SBATCH --job-name save_samples
#SBATCH --output ${DIFFSDA_LOGS_ROOT:-logs}/job-%x-%A-%a-%u.out
#SBATCH --gpus=rtx_6000:1
#SBATCH --cpus-per-task=6
#SBATCH --qos=normal
#SBATCH --array=1-5
#SBATCH --mem=60G

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda
source activate ${DIFFSDA_CONDA_ENV:-diffsda}

PYTHONPATH=.. python ../evaluation/faces/utils_aed_akd_faces.py --dataset ${DATASET:-vox1} --iter $SLURM_ARRAY_TASK_ID
