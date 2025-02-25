#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Call this script with 'srun' for a "quick" in-terminal run 
# or with 'sbatch' to submit independently of the session
# When running with 'srun' you have the pass the #SBATCH 
# arguments explicitely
# Availables partitions with GPUS at the AIP are 
# 'gpu' and 'debug'
 
# Modify the rest to comply to your configuration

source /opt/aconda3/etc/profile.d/conda.sh
conda activate py38-tf
ipython $*
