#!/bin/bash

################################################################################################
### Encode CelebV-HQ frames into VQ-f8ft latent representations.
### Submit as a Slurm job array (10 sections by default).
################################################################################################

#SBATCH --partition main
#SBATCH --time 7-00:00:00
#SBATCH --job-name prepare_latent_celebv
#SBATCH --output ${DIFFSDA_LOGS_ROOT:-logs}/job-%x-%A-%a-%u.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --qos=normal
#SBATCH --array=1-10
#SBATCH --mem=60G

echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda
source activate ${DIFFSDA_CONDA_ENV:-diffsda}

python celebv_to_latent.py --section=$SLURM_ARRAY_TASK_ID --input_size=256 --first_stage_model=vq8ft --face_crop
