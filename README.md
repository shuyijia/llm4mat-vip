# LLM 4 Materials for VIP @ GT

This package has been tested with the torchtune package at the specific commit identified by the following SHA (24 Feb 2025):

```
7b654ea00980778bc17773c7b1ab29af34d9af2b
```

In general, the latest main branch of torchtune should still be compatible. Thus, it is recommended to first follow the installation guide below. If you encounter any compatibility issues, try installing the version referenced above using the provided commit SHA:

```
# try the installation guide below before doing this
git clone https://github.com/pytorch/torchtune.git
cd torchtune
git checkout 7b654ea00980778bc17773c7b1ab29af34d9af2b
pip install -e .
```

## Installation
```
# 1. request a GPU on ICE
# e.g. request a single V100
salloc -N1 -t02:00:00 --gres=gpu:V100:1  --ntasks-per-node=4

# 2. load conda package manager
module load anaconda3

# create environment
conda create -n llmft python=3.11

# activate the environment
conda activate llmft

# install torch
pip install torch torchvision torchao

# install torchtune
git clone https://github.com/pytorch/torchtune.git
pip install -e torchtune/

# install this package
git clone https://github.com/shuyijia/llm4mat-vip.git
pip install -e llm4mat-vip

# install any additional packages you might need
pip install ase
```

## Dataset
`cd` into the `data/mp_20/` folder and unzip all zip files by doing
```
unzip '*.zip'
```

## Download LLMs
Go to https://github.com/pytorch/torchtune/tree/main/recipes/configs and choose the model you want to finetune. Look for the `.yaml` file that ends with `lora_single_device.yaml` for fine-tuning with LoRA on a single device. You can copy the `.yaml` file to the `configs` folder and modify it directly.

Single device here means training on a single GPU.

Look at the first few lines of your chosen yaml file, there should be a command on downloading that specific LLM weights. E.g.:
```
tune download meta-llama/Meta-Llama-3.1-8B-Instruct --output-dir [OUTPUT/DIR/TO/SAVE/THE/WEIGHTS] --ignore-patterns "original/consolidated.00.pth" --hf-token [YOUR/HUGGINGFACE/TOKEN]
```

## Fine-tuning
To fine-tune on an interactive node, do
```
tune run lora_finetune_single_device --config [PATH/TO/YOUR/YAML/FILE]
```

To submit the fine-tuning as a job to the slurm queue, modify the `submit.sh` script (e.g. you need to change the email address and put in the path to your `yaml` file). Save the changes and do
```
sbatch submit.sh
```

To view the progress of the submitted job, do
```
squeue --user [YOUR/GT/USERNAME]
```
To cancel a job, do
```
scancel [JOBID]
```
The job ID can be found by doing the `squeue --user [YOUR/GT/USERNAME]` command.

## Inference
To do inference, edit the inference yaml files with your saved checkpoint paths and do
```
tune run llm4materials/inference/infer.py --config [PATH/TO/YOUR/INFER/YAML/FILE]
```
Sample checkpoints have been provided in the current `configs/llama2-7b-inference.yaml` and `configs/llama3-8b-inference.yaml` files. You should be able to run them out of the box. 