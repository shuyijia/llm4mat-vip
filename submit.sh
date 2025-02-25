#!/bin/bash
#SBATCH -Jtorchtune-run                                 # Job name
#SBATCH -N1 --ntasks-per-node=4                         # Number of nodes and cores per node required
#SBATCH --mem-per-cpu=32G                               # Memory per core
#SBATCH -t16:00:00                                      # Duration of the job
#SBATCH -otorchtune-run.out                             # Combined output and error messages file
#SBATCH --gres=gpu:H100:1
#SBATCH --mail-type=BEGIN,END,FAIL                      # Mail preferences
#SBATCH --mail-user=[YOUR-EMAIL]@gatech.edu             # E-mail address for notifications

# load anaconda
module load anaconda3
# activate environment
conda activate llmft
# check GPU, will be logged to the output file
nvidia-smi
# run your script
tune run lora_finetune_single_device --config [PATH/TO/YOUR/YAML/FILE]